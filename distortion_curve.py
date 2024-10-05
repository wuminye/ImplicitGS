import torch
import numpy as np
import pillow_heif
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
import point_cloud_utils as pcu
import os
import glob
import zCurve as z 
from argparse import ArgumentParser, Namespace
import sys
import pdb
import json
import wandb




parser = ArgumentParser(description="Training script parameters")
parser.add_argument('-s', type=str, default="None")
parser.add_argument('-m', type=str, default="garden70")
parser.add_argument("--raw_points", action="store_true")
args = parser.parse_args(sys.argv[1:])

ttt = ''
if args.raw_points:
    ttt = '_raw'


wandb.init(
# set the wandb project where this run will be logged
    project="3dgs_compress",
# track hyperparameters and run metadata
    config=vars(args),
    name = 'RD_'+args.m.split('/')[-1]+ttt
)


data_points = []

def work(q1,q2,q3,raw_point=False):
    if raw_point:
        os.system(f'python scripts/compress_3dgs.py  -m {args.m}  --qp {q1} {q2} {q3} --raw_points')
    else:
        os.system(f'python scripts/compress_3dgs.py  -m {args.m}  --qp {q1} {q2} {q3}')
    if args.s == "None":
        os.system(f'python render.py  -m {args.m} --skip_train --eval --decompressed ')
    else:
        os.system(f'python render.py -s {args.s}  -m {args.m} --skip_train --eval --decompressed ')
    os.system(f'python metrics.py  -m {args.m}')

    with open(f'{args.m}/results.json', 'r') as file:
        data = json.load(file)
        psnr = data['ours_50000']['PSNR']
        ssim = data['ours_50000']['SSIM']
        lpips = data['ours_50000']['LPIPS']

    with open(f'{args.m}/compressed/result_[{q1}, {q2}, {q3}].json', 'r') as file:
        data = json.load(file)
        size = data['total']


    label = f"{args.m.split('/')[-1]}_[{q1} {q2} {q3}]"  
    if raw_point:
        label = f"{args.m.split('/')[-1]}_[{q1} {q2} {q3}]_raw"
    data_points.append({"psnr": psnr, "ssim": ssim, "lpips":lpips, "size":size, "label": label})
    table = wandb.Table(data=[[d["psnr"], d["ssim"],d["lpips"], d["size"],d["label"]] for d in data_points], columns=["psnr", "ssim", "lpips", "size", "label"])

    wandb.log({"scatter_plot_psnr": wandb.plot.scatter(table, "size", "psnr", title="RD_PSNR")})
    wandb.log({"scatter_plot_ssim": wandb.plot.scatter(table, "size", "ssim", title="RD_SSIM")})
    wandb.log({"scatter_plot_ssim": wandb.plot.scatter(table, "size", "lpips", title="RD_LPIPS")})

    wandb.log({"Data": table})




mipnerf_scenes = ['bonsai', 'counter','kitchen', 'room','stump','bicycle','garden','treehill','flowers']
db_scenes =  ['drjohnson', 'playroom']

# compression parameters for low and high profiles
qps = [[55,60,20],[90,75,40]]

# use different compression parameters for mipnerf and deepblending
for scene in db_scenes:
    if scene in args.m.split('/')[-1]:
        qps = [[45,45,10],[70,60,40]]
        break

for scene in mipnerf_scenes:
    if scene in args.m.split('/')[-1]:
        qps = [[70,60,40],[100,100,100]]
        break


for qp in qps:
    work(qp[0],qp[1],qp[2],raw_point=args.raw_points)


            

