"""
MERF Pipeline class.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
import gc

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.utils import profiler

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)

from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from nerfstudio.models.base_model import Model, ModelConfig
import numpy as np
import torch
from merf.coord import stepsize_in_squash, contract
from merf.grid_utils import calculate_grid_config, world_to_grid
from merf.merf_model import MERFModel,MERFModelConfig

from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

@dataclass
class MERFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: MERFPipeline)
    """target class to instantiate"""


class MERFPipeline(VanillaPipeline):

    # def __init__(
    #     self,
    #     config: MERFPipelineConfig,
    #     device: str,
    #     test_mode: Literal["test", "val", "inference"] = "val",
    #     world_size: int = 1,
    #     local_rank: int = 0,
    #     grad_scaler: Optional[GradScaler] = None,
    # ):
    #     # super().__init__(config, device)
    #     self.config = config
    #     self.test_mode = test_mode
    #     self.datamanager: DataManager = config.datamanager.setup(
    #         device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
    #     )
    #     self.datamanager.to(device)
    #     # TODO(ethan): get rid of scene_bounds from the model
    #     assert self.datamanager.train_dataset is not None, "Missing input dataset"

    #     self._model = config.model.setup(
    #         scene_box=self.datamanager.train_dataset.scene_box,
    #         num_train_data=len(self.datamanager.train_dataset),
    #         metadata=self.datamanager.train_dataset.metadata,
    #         device=device,
    #         grad_scaler=grad_scaler,
    #         pre_plane=self.datamanager.pre_plane,
    #     )
    #     self.model.to(device)

    #     self.world_size = world_size
    #     if world_size > 1:
    #         self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
    #         dist.barrier(device_ids=[local_rank])
 
    def baking_merf(self):
        if self.model.config.weight_mask == True:
            self.model.baking_merf_model_new(self.datamanager)
            #self.model.baking_merf_model(self.datamanager)
        else:
            self.model.baking_merf_model(self.datamanager)
    
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        if key == 'img':
                            val =val.clamp(0.0, 1.0)
                            Image.fromarray((val * 255).byte().cpu().numpy()).save(
                                output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                            )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        psnr_list = [metrics_dict["psnr"] for metrics_dict in metrics_dict_list]
        ssim_list = [metrics_dict["ssim"] for metrics_dict in metrics_dict_list]
        lpips_list = [metrics_dict["lpips"] for metrics_dict in metrics_dict_list]
        metrics_dict["psnr_list"] = psnr_list
        metrics_dict["ssim_list"] = ssim_list
        metrics_dict["lpips_list"] = lpips_list
        self.train()
        return metrics_dict

   