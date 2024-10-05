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




kk = 6.5
ks = [4,4,4]
parser = ArgumentParser(description="Training script parameters")
parser.add_argument('-m', type=str, default="garden70")
parser.add_argument("--raw_points", action="store_true")
parser.add_argument("--qp", nargs="+", type=int, default=[55,45,20])
args = parser.parse_args(sys.argv[1:])

qualities = args.qp
print("=====QP:",qualities)

name = args.m
os.makedirs(os.path.join(name, 'compressed'),exist_ok=True)
cpath = os.path.join(name, 'compressed')
path = f'{name}/chkpnt50000.pth'
#path = 'chkpnt30000.pth'
cpt = torch.load(path)
def tile_maker(feat_plane, h = 10000, w= 10000):
    image = torch.zeros(h,w)

    h,w = list(feat_plane.size())[-2:]


    x,y = 0,0
    for i in range(feat_plane.size(0)):
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "Tile_maker: not enough space, please increase the image resolution"

        image[x:x+h,y:y+w] = feat_plane[i,:,:]
        y = y + w
    return image
def quantize(data, nbits = 8, high_bound = kk, low_bound = -kk):
    nbits = 2**nbits-1
    data = np.copy(data)
    
    data[data<low_bound] = low_bound
    data[data>high_bound] = high_bound

    data = (data - low_bound)/(high_bound-low_bound)
    

    data = np.round(data *nbits)/nbits
    
    data = data *nbits

    return data


points = cpt[1]



properties_name = ['feat']
plane_name = ['xy_plane','xz_plane','yz_plane']




xyz = points.cpu()
xyz_min = xyz.min(dim=0)[0]
xyz_max=xyz.max(dim=0)[0]

ind_norm = (xyz - xyz_min) / (xyz_max - xyz_min) 


sorted_distances, indices_distances = torch.sort(ind_norm[:, 0])
ind_norm = ind_norm[indices_distances]

print(ind_norm.size())
fi_size = math.floor(math.sqrt(ind_norm.size(0))) + 1

point_image = torch.zeros(3,fi_size*fi_size)
for i in range(3):
    point_image[i,:ind_norm.size(0)] = ind_norm[:,i]
    

point_image = point_image.view(fi_size*3,fi_size).detach().numpy()

res = quantize(point_image,nbits=16, high_bound = 1, low_bound = 0).astype(np.uint16)


image = Image.fromarray(res)
image.save(f"{cpath}/points.png", format='PNG', compress_level=9) 




data_pts = (ind_norm*(2**16)).int().detach().numpy()
morton_codes = pcu.morton_encode(data_pts)
indexes = np.argsort(morton_codes).astype(np.int32)

grid_x, grid_y = torch.meshgrid( torch.linspace(0, fi_size,steps = fi_size),torch.linspace(0, fi_size,steps = fi_size), indexing='ij')
grid_x = grid_x.int().view(-1).numpy()
grid_y = grid_y.int().view(-1).numpy()

codes_2d = []
for i in range(grid_y.shape[0]):
     codes_2d.append(z.interlace(int(grid_x[i]),int(grid_y[i]), dims=2, bits_per_dim=16))
indexes_2d = np.argsort(codes_2d).astype(np.int32)

rres = []
for i in range(3):
    point_image_morton = np.zeros((fi_size*fi_size))


    point_image_morton[indexes_2d[:indexes.shape[0]].astype(np.int32)] = data_pts[indexes,i]
    point_image_morton = np.reshape(point_image_morton,(fi_size,fi_size))
    rres.append(point_image_morton)
rres = np.concatenate(rres,axis = 0).astype(np.uint16)
image = Image.fromarray(rres)
image.save(f"{cpath}/points_morton.png", format='PNG', compress_level=9) 




#----------------------------------------------------------------------




subplanes = [{},{},{}]
for indx in range(0,3):
    for i in properties_name:
        for indd,j in enumerate(plane_name):
            subplanes[indx][(i,j)] = cpt[4][f'_{i}.k0s.{indx}.{j}'][0].detach().cpu()
            mask = (subplanes[indx][(i,j)]<ks[indx]) & (subplanes[indx][(i,j)]>-ks[indx])
            print(i,j,subplanes[indx][(i,j)].min(),subplanes[indx][(i,j)].max(), (mask.sum()/mask.size(0)/mask.size(1)/mask.size(2)).item(),'%' )

    
    for ind, i in enumerate(properties_name):
        for indd,j in enumerate(plane_name):
            print(subplanes[indx][(i,j)].size(), qualities[indx])
            tiled_feature = tile_maker(subplanes[indx][(i,j)]).numpy()
            res = quantize(tiled_feature,nbits=16,high_bound=ks[indx],low_bound=-ks[indx]).astype(np.uint16)

            image = Image.fromarray(res)
        
        
            heif_file = pillow_heif.from_bytes(
                mode="L;16",
                size=(res.shape[1], res.shape[0]),
                data=bytes(res)
            )
            heif_file.save(f"{cpath}/{i},{j}_{indx}.heic", quality=qualities[indx])
            

mlps = []
for key in cpt[4].keys():
    if 'models.' in key:
        mlps.append(cpt[4][key].half().float())  # half-precision 
torch.save(tuple(mlps), f"{cpath}/mpls.pth")
        

def get_total_size_of_files(directory, file_type):
    # Build the pattern to search for files
    pattern = os.path.join(directory, f"*{file_type}")
    # Find all files matching the pattern
    files = glob.glob(pattern)
    # Calculate the total size
    total_size = sum(os.path.getsize(file) for file in files)
    return total_size




def untile_image(image,h,w,ndim):

    features = torch.zeros(ndim,h,w)
    x,y = 0,0
    for i in range(ndim):
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "untile_image: too many feature maps"
        features[i,:,:] = image[x:x+h,y:y+w]
        y = y + w
    return features

def unquantize(inputs,  nbits = 16, high_bound = kk, low_bound = -kk):
    nbits = 2**nbits-1
    data = inputs / nbits
    
    data = data*(high_bound-low_bound) + low_bound 
    return data
    
# ============== Points ===============================
heif_file = image_rec = Image.open(f'{cpath}/points_morton.png')
np_array = np.asarray(heif_file)

xyz_rec = unquantize(torch.tensor(np_array.astype(np.float32)),high_bound = 1, low_bound = 0).reshape(3,fi_size*fi_size)

rec_pts = torch.zeros_like(ind_norm)
for i in range(3):
    rec_pts[:,i] = xyz_rec[i,:ind_norm.size(0)]
print(torch.mean(torch.abs(ind_norm-rec_pts)))

xyz_rec = rec_pts*(xyz_max - xyz_min) + xyz_min
xyz_rec = xyz_rec[:xyz.size(0),:]

# ============== planes ===============================


subplanes_rec = [{},{},{}]
for indx in range(0,3):
    for ind, i in enumerate(properties_name):
        for indd,j in enumerate(plane_name):
            heif_file = pillow_heif.open_heif(f"{cpath}/{i},{j}_{indx}.heic",convert_hdr_to_8bit=False,bgr_mode=False)


            np_array = np.asarray(heif_file)

            

            np_array = torch.tensor(np_array[:,:].astype(np.float32))

            if len(np_array.size())>2:
                np_array = np_array[:,:,0]
            
            
            untiled = untile_image(np_array, h = subplanes[indx][(i,j)].size(1), w= subplanes[indx][(i,j)].size(2), ndim=subplanes[indx][(i,j)].size(0) )

            subplanes_rec[indx][(i,j)] = unquantize(untiled,nbits=16,high_bound=ks[indx],low_bound=-ks[indx])
            print('subplanes:', indx,j,'Errors:', torch.mean(torch.abs(subplanes[indx][(i,j)] - subplanes_rec[indx][(i,j)]))*ks[indx]*2)


cpt = torch.load(path)





for indx in range(0,3):
    for i in properties_name:
        for j in plane_name:
            cpt[4][f'_{i}.k0s.{indx}.{j}'][0] = subplanes_rec[indx][(i,j)]

if not args.raw_points:
    cpt[1].data = xyz_rec.cuda()
else:
    cpt[1].data = cpt[1].data.half().float()



torch.save(tuple(cpt), path+'_rec.pth')


total_size1 = get_total_size_of_files(cpath, '.heic')/1024/1024
print(f"Total size of all .heic files: {total_size1} MB")

total_size2 = get_total_size_of_files(cpath, 'points.png')/1024/1024
print(f"Total size of all png files: {total_size2} MB")



total_size3 = get_total_size_of_files(cpath, 'points_morton.png')/1024/1024
print(f"Total size of points_morton files: {total_size3} MB")

total_size4 = get_total_size_of_files(cpath, "mpls.pth")/1024/1024/2
print(f"Total size of MLP files: {total_size4} MB")



print("raw point size: ",points.size(0)*3*2/1024/1024)
print(name)
print('------------------')
print(f'TOTAL x_sorting:  {total_size1+total_size2+total_size4} MB')
print(f'TOTAL morton_sorting:  {total_size1+total_size3+total_size4} MB')
if args.raw_points:
    print(f'TOTAL raw_points:  {total_size1+total_size4 + points.size(0)*3*2/1024/1024} MB')
    total_size3 = points.size(0)*3*2/1024/1024
    total_size2 = points.size(0)*3*2/1024/1024
print(points.size())


res = {'heic':total_size1,'xsort':total_size2,'morton':total_size3,'MLP':total_size4,'total':min(total_size1+total_size2+total_size4,total_size1+total_size3+total_size4)}

with open(f'{cpath}/result_{str(qualities)}.json', 'w') as f:
    json.dump(res, f, indent=4)