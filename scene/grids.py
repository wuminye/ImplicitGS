
#
# Copyright (C) 2024, KU Leuven
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  minye.wu@kuleuven.be
#

import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class FakeQuantize(nn.Module):
    def __init__(self, scale = 5.0/127, zero_point=128, quant_min=0, quant_max=255):
        super(FakeQuantize, self).__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.quant_min = quant_min
        self.quant_max = quant_max

    def set_bits(self, n_bits):
        n_bits = 2**n_bits
        self.scale = 5.0/(n_bits/2-1)
        self.zero_point = n_bits/2
        self.quant_min = 0
        self.quant_max = n_bits-1


    def forward(self, x):
        x_int = torch.clamp(torch.floor(x / self.scale + self.zero_point), self.quant_min, self.quant_max)
        x_quant = (x_int - self.zero_point) * self.scale
        return x_quant

def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(**kwargs)
    elif type == 'TensoRFGrid':
        return TensoRFGrid(**kwargs)
    elif type == 'PlaneGrid':
        return PlaneGrid(**kwargs)
    elif type == 'DPlaneGrid':
        return DPlaneGrid(**kwargs)
    else:
        raise NotImplementedError




class PlaneGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max,  config, residual_mode = False):
        super(PlaneGrid, self).__init__()
        if 'factor' in config:
            self.scale = config['factor']
        else:
            self.scale = 2
            
        self.channels = channels
        self.world_size = world_size
        self.config = config
        self.residual_mode = residual_mode
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        X, Y, Z = world_size
        X = X*self.scale
        Y = Y*self.scale
        Z = Z*self.scale
        self.world_size = torch.tensor([X,Y,Z])
        R = self.channels //3
        Rxy = R
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y ]) * 0.1)
        self.xz_plane = nn.Parameter(torch.randn([1, R,  X, Z]) * 0.1)
        self.yz_plane = nn.Parameter(torch.randn([1, R,  Y, Z]) * 0.1)


        self.quant = FakeQuantize()
        self.quant.set_bits(12)



        print("Planes version activated !!!!!! ")

    
    def quant_all(self):
        self.xy_plane.data = self.quant(self.xy_plane.data)
        self.xz_plane.data = self.quant(self.xz_plane.data)
        self.yz_plane.data = self.quant(self.yz_plane.data)




    def compute_planes_feat(self, ind_norm, Q ):
        # Interp feature (feat shape: [n_pts, n_comp])

        xy_feat = F.grid_sample(self.xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
        xz_feat = F.grid_sample(self.xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
        yz_feat = F.grid_sample(self.yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T

        # Aggregate components
        if Q == 0:
            feat = torch.cat([
                xy_feat ,
                xz_feat ,
                yz_feat
            ], dim=-1)
        else:
            feat = torch.cat([
                xy_feat + torch.empty_like(xy_feat).uniform_(-0.5, 0.5) * Q,
                xz_feat + torch.empty_like(xz_feat).uniform_(-0.5, 0.5) * Q,
                yz_feat + torch.empty_like(yz_feat).uniform_(-0.5, 0.5) * Q 
            ], dim=-1)


        return feat       

    def forward(self, xyz, Q = 0, dir=None, center=None):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,-1,3)
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        ind_norm = torch.cat([ind_norm, torch.zeros_like(ind_norm[...,[0]])], dim=-1)

       
        if self.channels > 1:
            out = self.compute_planes_feat(ind_norm, Q=Q)
            out = out.reshape(*shape,self.channels)
        else:
            raise Exception("no implement!!!!!!!!!!")
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        X = X*self.scale
        Y=Y*self.scale
        Z = Z*self.scale

        self.xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X,Y], mode='bilinear', align_corners=True))
        self.xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X,Z], mode='bilinear', align_corners=True))
        self.yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y,Z], mode='bilinear', align_corners=True))

    def scale_volume_grid_value(self, new_world_size):
        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        X = X*self.scale
        Y=Y*self.scale
        Z = Z*self.scale
    
        xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X,Y], mode='bilinear', align_corners=True), requires_grad=False)
        xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X,Z], mode='bilinear', align_corners=True), requires_grad=False)
        yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y,Z], mode='bilinear', align_corners=True), requires_grad=False)

        return xy_plane, xz_plane, yz_plane

    def get_dim(self):
        return self.channels

    def total_variation_add_grad(self, w):
        '''Add gradients by total variation loss in-place'''
        wx = wy= wz = w
        loss = wx * F.smooth_l1_loss(self.xy_plane[:,:,1:], self.xy_plane[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.xy_plane[:,:,:,1:], self.xy_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.xz_plane[:,:,1:], self.xz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.xz_plane[:,:,:,1:], self.xz_plane[:,:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.yz_plane[:,:,1:], self.yz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.yz_plane[:,:,:,1:], self.yz_plane[:,:,:,:-1], reduction='sum') 
        loss /= 6
        loss.backward()


    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}, n_comp={self.channels //3}'



