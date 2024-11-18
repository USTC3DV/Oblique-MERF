import dataclasses
from dataclasses import dataclass, field
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig,VanillaPipeline
from nerfstudio.plugins.types import MethodSpecification
from merf.merf_model import MERFModel, MERFModelConfig
from merf.engine.merf_trainer import MERFTrainerConfig
from merf.merf_datamanager import MerfDataParser,MerfDataParserConfig
from merf.pipeline.merf_pipeline import MERFPipeline, MERFPipelineConfig

merf_config = MERFTrainerConfig(
    method_name="merf-ns",
    viewer=ViewerConfig(
        relative_log_filename='viewer_log_filename.txt',
        websocket_port=None,
        websocket_port_default=7007,
        websocket_host='0.0.0.0',
        num_rays_per_chunk=4,
        max_num_display_images=512,
        quit_on_train_completion=False,
        image_format='jpeg',
        jpeg_quality=90
    ),
    pipeline=MERFPipelineConfig(
        _target=MERFPipeline,
        datamanager=VanillaDataManagerConfig(
            # _target=RayPruningDataManager,
            dataparser=MerfDataParserConfig(
            is_transform = True,
            downscale_factor=1,
            train_split_fraction=0.99,
            orientation_method='vertical',
            center_method='poses',
            scene_scale=1.0,
            scale_factor=1.0,
            z_limit_min=-1.0,
            z_limit_max=1.0,
            scale_1 = 1.0
            ),
            eval_num_rays_per_batch=16384,
            train_num_rays_per_batch=32768,
        ),
        model=MERFModelConfig(_target=MERFModel,
                               alpha_threshold_param=(10000, 30000, 5e-4, 5e-3, 20000),
                                occ_plane_size=512,
                                occ_plane_eps=0.004,
                                occ_loss_mult_param=(10000, 2000, 1e-4, 1.5, 2e-1),
                                occ_start_step=10000,
                                occ_end_step=80001,
                                sparse_grid_resolution=512,
                                triplane_resolution=2048,
                                weight_mask=True,
                                spatial_distortion=False,
                                occ_density_loss_mult=0.05,
                                distortion_loss_mult=0.01,
                                num_proposal_samples_per_ray=(128, 64),
                                num_nerf_samples_per_ray=32,
                                occ_specular_mult = 0.1,
                                occ_celoss_mult = 0.001,
                                patch_w=256,
                                patch_h=128,
                                log2_hashmap_size=22,
                                num_random_samples=2**14
                              ),
    ),
    max_num_iterations=80000,
    steps_per_save=10000,
    steps_per_eval_batch=2000,
    steps_per_eval_image=10000,
    steps_per_eval_all_images=79999,
    optimizers={ 
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=100000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=100000),
        },
        "deferred_mlp": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=100000),
        },
        "occ_planes": {
            "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=100000),
        }
    },
    gradient_accumulation_steps = 1,
)
MERFNS = MethodSpecification(
    config=merf_config, description="Unofficial implementation of MERF paper"
)
