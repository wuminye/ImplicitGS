#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr 
#

#
# Copyright (C) 2024, KU Leuven
# All rights reserved.
#
#
# For inquiries contact  minye.wu@kuleuven.be
#




import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from .grids import PlaneGrid
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pdb
import time

from utils.entropy_models import  Entropy_gaussian



class Conctractor(nn.Module):
    def __init__(self, xyz_min, xyz_max, enable = True):
        super().__init__()
        self.enable = enable
        if not self.enable:
            print('**Disable Contractor**')
        self.register_buffer('xyz_min', xyz_min)
        self.register_buffer('xyz_max', xyz_max)

    def decontracte(self, xyz): 
        if not self.enable:
            raise Exception("Not implement")

        mask = torch.abs(xyz) > 1.0
        res = xyz.clone()
        signs = (res <0) & (torch.abs(res)>1.0)
        res[mask] = 1.0/(1.0- (torch.abs(res[mask])-1)) 
        res[signs] *= -1
        res = res * (self.xyz_max-self.xyz_min) /2 + (self.xyz_max+self.xyz_min) /2

        return res
    
    def contracte(self, xyz):

        indnorm = (xyz-self.xyz_min)*2.0 / (self.xyz_max-self.xyz_min) -1
        if self.enable:
            mask = torch.abs(indnorm)>1.0
            signs = (indnorm <0) & (torch.abs(indnorm)>1.0)
            indnorm[mask] = (1.0- 1.0/torch.abs(indnorm[mask])) +1.0
            indnorm[signs] *=-1
        return indnorm

class FeaturePlanes(nn.Module):
    def __init__(self, world_size, xyz_min, xyz_max, feat_dim = 24, mlp_width = [168], out_dim=[53], subplane_multiplier=1):
        super(FeaturePlanes, self).__init__()
        
        self.world_size, self.xyz_min, self.xyz_max = world_size, xyz_min, xyz_max

        self.activate_level = 0
        self.num_levels = 3
        self.level_factor = 0.5

        t_ws = torch.tensor(world_size)

        self.k0s =  torch.nn.ModuleList()

        for i in range(self.num_levels):
            cur_ws = (t_ws*self.level_factor**(self.num_levels-i-1)).cpu().int().numpy().tolist()
            self.k0s.append(PlaneGrid(feat_dim, cur_ws, xyz_min, xyz_max,config={'factor':1}))
            print('Create Planes @ ', cur_ws)


        self.models = torch.nn.ModuleList()

        mlp_width = [mlp_width[0],mlp_width[0],mlp_width[0]] 
        out_dim = [out_dim[0],out_dim[0],out_dim[0]]

        for i in range(self.num_levels):
            self.models.append(nn.Sequential(
                                nn.Linear(self.k0s[i].get_dim(), mlp_width[i]),
                                nn.ReLU(),
                                nn.Linear(mlp_width[i], mlp_width[i]),
                                nn.ReLU(),
                                nn.Linear(mlp_width[i], out_dim[i])
                                ))


    def forward(self, x, Q=0):
        # Pass the input through k0

        level_features = []

        for i in range(self.activate_level + 1):
            feat = self.k0s[i](x , Q)
            level_features.append(feat)

        res = []
        cnt =0
        for m,feat in zip(self.models,level_features):
            rr = m(feat)
            res.append(rr)
            cnt = cnt + 1
            if cnt>self.activate_level:
                break
        
        return sum(res)
        
        
        
        


        
class GaussianLearner(nn.Module):
    def __init__(self, model_params, xyz_min = [-2, -2, -2], xyz_max=[2, 2, 2] ):
        super(GaussianLearner, self).__init__()

        self.Q0 = 0.03
        self.xyz_min = torch.tensor(xyz_min).cuda()
        self.xyz_max = torch.tensor(xyz_max).cuda()

        self.world_size = [model_params.plane_size]*3
        self.max_step = 6
        self.current_step = 0

        self._feat = FeaturePlanes(world_size=self.world_size, xyz_min = self.xyz_min, xyz_max= self.xyz_max,
                                    feat_dim = model_params.num_channels, mlp_width = [model_params.mlp_dim], out_dim=[35], subplane_multiplier=model_params.subplane_multiplier )  # 27,4,3,1

        self.register_buffer('opacity_scale', torch.tensor(10))
        self.opacity_scale = self.opacity_scale.cuda()

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()


    def activate_plane_level(self):
        self._feat.activate_level +=1
        print('******* Plane Level to:', self._feat.activate_level)


    def inference(self, xyz):
        inputs = xyz.cuda().detach()
        
        tmp  = self._feat(inputs, self.Q0)
        features = tmp[:,:27]
        rotations = tmp[:,27:27+4]
        scale = tmp[:,31:31+3]
        opacity = tmp[:,34:]
        scale = torch.sigmoid(scale)

        return opacity*10, scale, features, rotations

    def tv_loss(self, w):
        for level in range(self._feat.activate_level+1):
            factor = 1.0
            self._feat.k0s[level].total_variation_add_grad(w*((0.5)**(2-level)))
            

    def calc_sparsity(self):

        plane = self._feat
        res = 0
        for level in range(self._feat.activate_level+1):
  
            factor = 1.0
            
            for data in [plane.k0s[level].xy_plane, plane.k0s[level].xz_plane, plane.k0s[level].yz_plane]:
                l1norm = torch.mean(torch.abs(data))
                res += l1norm * ((0.4)**(2-level)) * factor

        return res / ((self._feat.activate_level+1)*3)

        





class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def setup_contractor(self,center,length, contractor):
        center = torch.tensor(center)
        length = torch.tensor(length)
        self.contractor = Conctractor(xyz_min=center-length*self.bbox_scale/2, xyz_max=center+length*self.bbox_scale/2, enable = contractor)
        self.contractor = self.contractor.cuda()




    def __init__(self, sh_degree, model_params =None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)


        self._features_dc = torch.empty(0)
        self._scaling = torch.empty(0)
        self._opacity = torch.empty(0)

        self.feat_planes = GaussianLearner(model_params).cuda()



        self.deform = False

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.magic_k = False
        self.enable_net = False

        self.bbox_scale = model_params.bbox_scale

        self.setup_contractor(model_params.scene_center, model_params.scene_length, model_params.contractor )


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._rotation,
            self._scaling,
            self.feat_planes.state_dict(),
            self.contractor.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def scale_grid(self):
        if not self.feat_planes.scale_grid():
            self.training_setup(self.training_args)

        

    @property
    def get_scaling(self):
        x = self._scaling
        return self.scaling_activation(x)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_scaling_net(self):
        x = self._scaling_net
        return self.scaling_activation(x)
    
    @property
    def get_rotation_net(self):
        return self.rotation_activation(self._rotation_net)
    
    
    @property
    def get_xyz(self):
            return self._xyz 
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_net(self):
        features_dc = self._features_dc_net
        features_rest = self._features_rest_net
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)


    @property
    def get_opacity_net(self):
        return self.opacity_activation(self._opacity_net)

    def build_properties(self, data, visible ):

        tmp = torch.ones([self._xyz.size(0)]+list(data.size()[1:]),device = data.device)*-5
        tmp[visible] = data
        return tmp

    def activate_plane_level(self):
        self.feat_planes.activate_plane_level()
        self.training_setup(self.training_args)

    def update_contractor(self):
        points = self.get_xyz.detach()

        center = torch.mean(points,dim=0)
        length = (torch.max(points,dim=0)[0] - torch.min(points,dim=0)[0])*1.1

        self.setup_contractor(center.cpu().tolist(),length.cpu().tolist(), False)
        print('scene_center:',center.cpu().tolist(),'scene_length',length.cpu().tolist())
    

    def inference_gaussians(self, visible = None):
        points = self.get_xyz

        if visible is None:
            visible = torch.ones(points.size(0),device = points.device).bool()
        
        opacity, scales, features,rotations = self.feat_planes.inference(self.contractor.contracte(points.detach()[visible]))
    
        scales = (scales-1)*5-2
        features = features.view(features.size(0),(self.max_sh_degree + 1) ** 2,3)
        feature_dc = features[:,0:1,:]
        feature_rest = features[:,1:,:]


        self._opacity_net = self.build_properties(opacity,visible)
        self._scaling_net = self.build_properties(scales,visible)
        self._rotation_net = self.build_properties(rotations,visible)
        self._features_dc_net = self.build_properties(feature_dc,visible)
        self._features_rest_net = self.build_properties(feature_rest,visible)


        return points

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance_net(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling_net, scaling_modifier, self._rotation_net)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))


        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")



    def training_setup(self, training_args):

        self.training_args = training_args
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")


        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        for i in range(3):
            if i == self.feat_planes._feat.activate_level:
                l.append( {'params': self.feat_planes._feat.k0s[i].parameters(), 'lr': 0.01, 'name': 'feat_planes%d'%i})
                l.append( {'params': self.feat_planes._feat.models[i].parameters(), 'lr': 1e-4, 'name': 'fp_mlp_f%d'%i})
            else:
                l.append( {'params': self.feat_planes._feat.k0s[i].parameters(), 'lr': 0.001, 'name': 'feat_planes%d'%i})
                l.append( {'params': self.feat_planes._feat.models[i].parameters(), 'lr': 1e-5, 'name': 'fp_mlp_f%d'%i})


        self.optimizer = torch.optim.Adam(l, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.planes_scheduler_args = get_expon_lr_func(lr_init=0.01,
                                                    lr_final=0.005,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.planesmlp_scheduler_args = get_expon_lr_func(lr_init=1e-4,
                                                    lr_final=5e-5,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

                

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        print('reset_opacity.......')
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  len(group["params"]) != 1 :
                continue
            if group["name"]=='Qs' or group["name"]=='sigmas':
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  len(group["params"]) != 1 :
                continue
            if group["name"]=='Qs' or group["name"]=='sigmas':
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  len(group["params"]) != 1 :
                continue
            if group["name"]=='Qs' or group["name"]=='sigmas':
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def clone_and_grow(self):

        
        new_xyz = self._xyz + torch.randn(self._xyz.size(), device = self._xyz.device)*0.1
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity
        new_scaling = self._scaling
        new_rotation = self._rotation

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        with torch.no_grad():
            self.inference_gaussians()
        self.densify_and_split(grads, max_grad, extent)
        with torch.no_grad():
            self.inference_gaussians()

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def graph_downsampling(self, rate, mode= "random"):
        pts = self.get_xyz
        pts = pts.detach().cpu().numpy()
        num_pts = pts.shape[0]
        mask = torch.ones((num_pts), device="cuda", dtype=bool)
        t1 = time.time()
        print("Graph Downsampling Processing, points number before sampling: ", num_pts)
        if mode == "random":
            idxs = np.random.choice(num_pts, int(np.floor(num_pts * rate)), replace=False)
        idxs = torch.from_numpy(idxs).long().cuda()
        mask[idxs] = 0
        self.prune_points(mask)
        
        torch.cuda.empty_cache()
        print("Graph Downsampling Processed, points number after sampling: ", self.get_xyz.shape[0], "Time: ", time.time() - t1, "seconds")
  


    def prune_points_m(self, min_opacity):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1