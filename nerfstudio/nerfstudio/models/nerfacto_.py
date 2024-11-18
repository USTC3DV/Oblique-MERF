# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields_occ import HashMLPDensityFieldOcc
from nerfstudio.fields.nerfacto_field_occ import NerfactoFieldOcc
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler, Sampler, UniformLinDispPiecewiseSampler,PDFSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

from merf.coord import stepsize_in_squash
def sparsity_loss(random_positions, random_viewdirs, density, voxel_size):
  step_size = stepsize_in_squash(
      random_positions, random_viewdirs, voxel_size
  )
  return 1.0 - torch.exp(-step_size.unsqueeze(-1) * density).mean()

from typing import Any, Callable, List, Optional, Protocol, Tuple, Union
class ProposalNetworkSamplerOcc(Sampler):
    """Sampler that uses a proposal network to generate samples.

    Args:
        num_proposal_samples_per_ray: Number of samples to generate per ray for each proposal step.
        num_nerf_samples_per_ray: Number of samples to generate per ray for the NERF model.
        num_proposal_network_iterations: Number of proposal network iterations to run.
        single_jitter: Use a same random jitter for all samples along a ray.
        update_sched: A function that takes the iteration number of steps between updates.
        initial_sampler: Sampler to use for the first iteration. Uses UniformLinDispPiecewise if not set.
    """

    def __init__(
        self,
        num_proposal_samples_per_ray: Tuple[int, ...] = (64,),
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 2,
        single_jitter: bool = False,
        update_sched: Callable = lambda x: 1,
        initial_sampler: Optional[Sampler] = None,
    ) -> None:
        super().__init__()
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")

        # samplers
        if initial_sampler is None:
            self.initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            self.initial_sampler = initial_sampler
        self.pdf_sampler = PDFSampler(include_original=False, single_jitter=single_jitter)

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def set_anneal(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._anneal = anneal

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self._steps_since_update += 1

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
        occ_planes: OccPlane = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                mask_to_train = None
                if occ_planes is not None:
                    weights_mask, mask_to_train=occ_planes.get_weight_mask_two(ray_samples)
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](ray_samples.frustums.get_positions(), mask_to_train=mask_to_train)
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.frustums.get_positions(), mask_to_train=mask_to_train)
                weights = ray_samples.get_weights(density)
                if occ_planes is not None:
                    if weights_mask is not None:
                        weights = weights * weights_mask
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list

from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
class OccPlane(nn.Module):

    def __init__(
        self,
        aabb: Float[Tensor, "2 3"],
        plane_size: float = 1024,
        plane_eps: float = 0.004,
        start_step: int = 30000,
        end_step: int = 250000,
        spatial_distortion: bool = True,
    ) -> None:
        super().__init__()
        self.plane_size = plane_size
        self.plane_eps = plane_eps
        # self.register_buffer("plane_size", plane_size)
        # self.register_buffer("plane_eps", plane_eps)
        # self.bound = max(aabb[1][1], aabb[1][0])
        if spatial_distortion is not None:
            self.bound = 2.0
        else:
            self.bound = 1.0
        self.aabb = aabb
        self.spatial_distortion = spatial_distortion
        self.start_step = start_step
        self.end_step = end_step
        plane_min = aabb[0][2]
        plane_max = aabb[1][2]
        self.occ_plane_min = nn.Parameter(plane_min * torch.ones(int(self.plane_size) ** 2, dtype=torch.float32), requires_grad=True)
        self.occ_plane_max = nn.Parameter(plane_max * torch.ones(int(self.plane_size) ** 2, dtype=torch.float32), requires_grad=True)

        self._step = 0
        self.loss_mult = 0.0
        self.weight_thred = 0.0
        
        self.sample1 = 0
        self.sample2 = 0
        self.sample_all = 1

    def step_occ(self, step):
        self._step = step
        
        if self.sample2>0:
            self.sample_all = self.sample1 / float(self.sample2)
        self.sample1 = 0
        self.sample2 = 0
    
    def set_weight_thred(self, weight_thred):
        self.weight_thred = weight_thred
    
    def set_loss_mult(self, loss_mult):
        self.loss_mult = loss_mult
    
    def get_weight_mask(
        self,
        ray_samples: RaySamples,
        weights: Float[Tensor, "*bs num_samples"],
        notprop: bool=True,
    ) -> Float[Tensor, "*bs num_samples"]:
        assert weights is not None
        assert ray_samples is not None
        
        add_mask = self._step > self.start_step

        if add_mask:
        
            # Make sure the tcnn gets inputs between 0 and 1.
            if self.spatial_distortion is not None:
                if self.gaussian_samples_in_prop or notprop:
                    positions = self.spatial_distortion(ray_samples.frustums.get_gaussian_samples())
                else:
                    positions = self.spatial_distortion(ray_samples.frustums.get_positions()) 
            else:
                if self.gaussian_samples_in_prop or notprop:
                    positions = ray_samples.frustums.get_gaussian_samples()
                    # positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_gaussian_samples(), self.aabb)
                else:
                    positions = ray_samples.frustums.get_positions()
                    # positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

            xyzs_temp = positions
            weights = weights.squeeze(-1)

            prefix = xyzs_temp.shape[:-1]
            x_index = torch.clamp(torch.floor((xyzs_temp[...,:1] + self.bound) * self.plane_size * 0.5 / self.bound), min=0, max=self.plane_size -1)
            y_index = torch.clamp(torch.floor((xyzs_temp[...,1:2] + self.bound) * self.plane_size * 0.5 / self.bound), min=0, max=self.plane_size -1)
                
            xy_index = (x_index * self.plane_size + y_index).long().reshape(-1)
            zz_min = self.occ_plane_min[xy_index].reshape(*prefix)
            zz_max = self.occ_plane_max[xy_index].reshape(*prefix)
            weights_mask = torch.ones_like(weights, device=weights.device, dtype=weights.dtype)
            zz = xyzs_temp[..., -1].squeeze(-1)
            weights_mask[zz < zz_min] = 0.0
            weights_mask[zz > zz_max] = 0.0
            if self._step <= self.end_step:
                # method 1
                index1 = (zz < zz_min + self.plane_eps) * (zz > zz_min) * (zz < zz_max - self.plane_eps)
                weights_mask[index1] = (zz[index1] - zz_min[index1])**2 / self.plane_eps**2
                index2 = (zz > zz_max - self.plane_eps) * (zz < zz_max) * (zz > zz_min + self.plane_eps)
                weights_mask[index2] = (-zz[index2] + zz_max[index2])**2 / self.plane_eps**2
                # method 2
                # plane_eps_ = (0.02 * (zz_max - zz_min)).detach()
                # index1 = (zz < zz_min + plane_eps_) * (zz > zz_min) * (zz < zz_max - plane_eps_)
                # weights_mask[index1] = (zz[index1] - zz_min[index1])**0.5 / plane_eps_[index1]**0.5
                # index2 = (zz > zz_max - plane_eps_) * (zz < zz_max) * (zz > zz_min + plane_eps_)
                # weights_mask[index2] = (-zz[index2] + zz_max[index2])**0.5 / plane_eps_[index2]**0.5
                # method 3
                # z_mean = (zz_min + zz_max) / 2.0
                # z_var = (zz_max - zz_min) / 2.0
                # z_gauss = torch.exp(-(zz - z_mean)**2 / (2* z_var**2))
                # z_sigmoid = 1.0 / (1 + torch.exp(-100.0*(z_gauss - torch.exp(torch.tensor([0.5])) )/ (2*3.14* z_var)))
                # index1 = (zz > zz_min) * (zz < zz_max)
                # weights_mask[index1] = z_sigmoid[index1]
            weights_mask.nan_to_num_(0)
            weights = weights * weights_mask
            
            weights = weights.unsqueeze(-1)

        return weights
    
    def get_weight_mask_two(
        self,
        ray_samples: RaySamples,
    ) -> Float[Tensor, "*bs num_samples"]:
        assert ray_samples is not None
        
        add_mask = self._step > self.start_step
        
        weights_mask = None
        mask_to_train = None

        if add_mask:
        
            # Make sure the tcnn gets inputs between 0 and 1.
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(ray_samples.frustums.get_positions()) 
            else:
                positions = ray_samples.frustums.get_positions()

            xyzs_temp = positions

            prefix = xyzs_temp.shape[:-1]
            x_index = torch.clamp(torch.floor((xyzs_temp[...,:1] + self.bound) * self.plane_size * 0.5 / self.bound), min=0, max=self.plane_size -1)
            y_index = torch.clamp(torch.floor((xyzs_temp[...,1:2] + self.bound) * self.plane_size * 0.5 / self.bound), min=0, max=self.plane_size -1)
                
            xy_index = (x_index * self.plane_size + y_index).long().reshape(-1)
            zz_min = self.occ_plane_min[xy_index].reshape(*prefix)
            zz_max = self.occ_plane_max[xy_index].reshape(*prefix)
            weights_mask = torch.ones(prefix, device=xyzs_temp.device, dtype=xyzs_temp.dtype)
            mask_to_train = torch.ones(prefix, device=xyzs_temp.device, dtype=torch.bool)
            zz = xyzs_temp[..., -1].squeeze(-1)
            weights_mask[zz < zz_min] = 0.0
            weights_mask[zz > zz_max] = 0.0
            mask_to_train[zz < zz_min] = False
            mask_to_train[zz > zz_max] = False
            if self._step <= self.end_step:
                # method 4
                index1_ = (zz < zz_min + self.plane_eps) * (zz > zz_min) * (zz < zz_max - self.plane_eps)
                weights_mask[index1_] = (zz[index1_] - zz_min[index1_])**2 / self.plane_eps**2
                # index1 = (zz > zz_min + self.plane_eps) * (zz < zz_max - self.plane_eps*3.0) * (zz < zz_min + self.plane_eps*3.0)
                # index1 = (zz > zz_min + self.plane_eps) * (zz < zz_max - self.plane_eps) * (zz < (zz_min + zz_max)/2.0)
                # weights_mask[index1] = (zz[index1] - zz_min[index1])**2 / self.plane_eps**2 - ((zz[index1] - zz_min[index1])**2 / self.plane_eps**2).detach() + 1.0
                
                index2_ = (zz > zz_max - self.plane_eps) * (zz < zz_max) * (zz > zz_min + self.plane_eps)
                weights_mask[index2_] = (-zz[index2_] + zz_max[index2_])**2 / self.plane_eps**2
                # index2 = (zz < zz_max - self.plane_eps) * (zz > zz_min + self.plane_eps*3.0) * (zz > zz_max - self.plane_eps*3.0)
                # index2 = (zz < zz_max - self.plane_eps) * (zz > zz_min + self.plane_eps) * (zz > (zz_max + zz_min)/2.0)
                # weights_mask[index2] = (-zz[index2] + zz_max[index2])**2 / self.plane_eps**2 - ((-zz[index2] + zz_max[index2])**2 / self.plane_eps**2).detach() + 1.0
            weights_mask.nan_to_num_(0)
            
            weights_mask = weights_mask.unsqueeze(-1)
            mask_to_train = mask_to_train.unsqueeze(-1)

            self.sample1 += (torch.sum(mask_to_train, dim=-1)).float().mean().item()
            self.sample2 += mask_to_train.shape[-1]
        
        return weights_mask, mask_to_train
    
    def get_occ_loss(self) -> float:
        if self._step > self.end_step:
            return 0.0
        L2_mean = nn.MSELoss()
        # method 1
        index = (self.occ_plane_max - self.occ_plane_min) > 0.00001
        occ_loss = L2_mean(self.occ_plane_max[index], self.occ_plane_min[index])
        # method 2
        # index = torch.where((self.occ_plane_max - self.occ_plane_min) > 0.00001, 1, 0).detach()
        # occ_loss = L2_mean(self.occ_plane_max*index, self.occ_plane_min*index)
        return occ_loss * self.loss_mult
    
    def get_occ_l1(self) -> float:
        occ_l1 = (self.occ_plane_max - self.occ_plane_min).mean()
        return occ_l1

@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = True
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    """params for occplane"""
    weight_mask : bool = True
    occ_start_step : int = 5000
    occ_end_step : int = 30000
    occ_plane_size : float = 512
    occ_plane_eps : float = 0.004
    occ_loss_mult_param : Tuple[int, int, float, float, float] = field(
      default_factory=lambda: (5000, 1000, 1e-5, 1.25, 1e-2)
    )#(start_iters, maintain_step, star_value, mult_value, end_value)
    occ_density_loss_mult : float = 0.01
    occ_celoss_mult: float = 0.0
    add_weight_thred: bool = False
    alpha_threshold_param:Tuple[int, int, float, float] = field(
      default_factory=lambda: (2500, 7500, 5e-4, 5e-3, 5000)
    ) 
    """alpha culling in MERF:(start iters, end iters, start culling value, end culling value)"""
    sparsity_loss_mult: float = 0.0
    """Sparsity loss multiplier."""


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = NerfactoFieldOcc(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityFieldOcc(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityFieldOcc(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # occplane
        if self.config.weight_mask:
            self.occ_planes = OccPlane(
                self.scene_box.aabb,
                plane_size=self.config.occ_plane_size,
                plane_eps=self.config.occ_plane_eps,
                start_step=self.config.occ_start_step,
                end_step=self.config.occ_end_step,
                spatial_distortion=scene_contraction,
            )
            self.occ_density_loss_mult = 0.0
            self.occ_celoss_mult = 0.0
        
        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSamplerOcc(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        # self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        self.collider = AABBBoxCollider(scene_box=self.scene_box,near_plane=self.config.near_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        if self.config.weight_mask:
            param_groups["occ_planes"] = list(self.occ_planes.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        
        def set_alpha_culling_value(step):
                
            def log_lerp(t, v0, v1):
                """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
                if v0 <= 0 or v1 <= 0:
                    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
                lv0 = np.log(v0)
                lv1 = np.log(v1)
                return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)
            
            if step < self.config.alpha_threshold_param[0]:
                alpha_thres = 0.0
            elif step > self.config.alpha_threshold_param[1]:
                alpha_thres = self.config.alpha_threshold_param[3]
            elif step > self.config.alpha_threshold_param[4]: 
                t = (step - self.config.alpha_threshold_param[4]) / (self.config.alpha_threshold_param[1] - self.config.alpha_threshold_param[4])
                alpha_thres = log_lerp(t, self.config.alpha_threshold_param[2], self.config.alpha_threshold_param[3])
            else:
                t =  (step - self.config.alpha_threshold_param[0]) / (self.config.alpha_threshold_param[4] - self.config.alpha_threshold_param[0])
                alpha_thres = t * self.config.alpha_threshold_param[2]
            
            self.field.set_alpha_threshold(alpha_thres)
            if self.config.add_weight_thred:
                self.occ_planes.set_weight_thred(alpha_thres)
        
        callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_alpha_culling_value,
                )
            )
        
        if self.config.weight_mask:
            def set_occ_loss_mult(step):
                if step > self.config.occ_end_step:
                    self.occ_density_loss_mult = 0
                elif step > self.config.occ_loss_mult_param[0]:
                    self.occ_density_loss_mult = self.config.occ_density_loss_mult
                    # if step > 50000:
                    self.occ_celoss_mult = self.config.occ_celoss_mult
                    step_since_add = step - self.config.occ_loss_mult_param[0]
                    occ_loss_mult_ = self.config.occ_loss_mult_param[2] * self.config.occ_loss_mult_param[3] ** (step_since_add // self.config.occ_loss_mult_param[1])
                    self.occ_planes.set_loss_mult(min(occ_loss_mult_, self.config.occ_loss_mult_param[4]))
                
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_occ_loss_mult,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.occ_planes.step_occ,
                )
            )
        
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        if self.config.weight_mask:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns, occ_planes=self.occ_planes)
        else:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        mask_to_train = None
        if self.config.weight_mask:
            weights_mask, mask_to_train = self.occ_planes.get_weight_mask_two(ray_samples)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals, mask_to_train=mask_to_train)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        if self.config.weight_mask:
            if weights_mask is not None:
                weights = weights * weights_mask
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        if self.config.weight_mask:
            metrics_dict["occL1"] = self.occ_planes.get_occ_l1()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            if self.config.weight_mask:
                loss_dict["occ_loss"] = self.occ_planes.get_occ_loss()
                if self.config.sparsity_loss_mult > 0.0 :
                    # sample points
                    num_random_samples = 2**14
                    random_positions = self.scene_box.aabb[0].min() + (self.scene_box.aabb[1].max() - self.scene_box.aabb[0].min()) * torch.rand(num_random_samples, 3).to(self.device)
                    random_viewdirs = torch.normal(mean=0, std=1, size=(num_random_samples, 3)).to(self.device)
                    random_viewdirs /= torch.norm(random_viewdirs, dim=-1, keepdim=True)
                    density = self.field.get_density_only(random_positions)
                    loss_dict["sparsity_loss"] = self.config.sparsity_loss_mult * sparsity_loss(random_positions, random_viewdirs, density, self.occ_planes.plane_size)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
