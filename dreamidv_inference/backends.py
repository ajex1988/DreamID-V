from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

from .config import InferenceConfig
from .preprocessors import ConditioningAssets


@dataclass
class RuntimeContext:
    rank: int
    world_size: int
    local_rank: int
    device: int
    distributed: bool


def init_logging(rank: int) -> None:
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def initialize_runtime(config: InferenceConfig) -> RuntimeContext:
    import torch
    import torch.distributed as dist

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    init_logging(rank)

    if config.model.offload_model is None:
        config.model.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {config.model.offload_model}."
        )

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        if config.model.t5_fsdp or config.model.dit_fsdp:
            raise ValueError("t5_fsdp and dit_fsdp are not supported in non-distributed mode.")
        if config.model.ulysses_size > 1 or config.model.ring_size > 1:
            raise ValueError("ulysses_size and ring_size are not supported in non-distributed mode.")

    if config.model.ulysses_size > 1 or config.model.ring_size > 1:
        if config.model.ulysses_size * config.model.ring_size != world_size:
            raise ValueError("ulysses_size * ring_size must equal world_size.")
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=config.model.ring_size,
            ulysses_degree=config.model.ulysses_size,
        )

    return RuntimeContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=local_rank,
        distributed=dist.is_initialized(),
    )


def barrier(runtime: RuntimeContext) -> None:
    if runtime.distributed:
        import torch.distributed as dist

        dist.barrier()


class BaseBackend:
    needs_pose = True

    def __init__(self, config: InferenceConfig, runtime: RuntimeContext):
        self.config = config
        self.runtime = runtime
        self.cfg = None
        self.size = None
        self.pipeline = None
        self.cache_video = None

    def generate_chunk(self, assets: ConditioningAssets, frame_num: int):
        raise NotImplementedError

    def save(self, video_tensor, save_file: str) -> None:
        self.cache_video(
            tensor=video_tensor[None],
            save_file=save_file,
            fps=self.cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )


class StandardBackend(BaseBackend):
    needs_pose = True

    def __init__(self, config: InferenceConfig, runtime: RuntimeContext):
        super().__init__(config, runtime)
        import dreamidv_wan
        from dreamidv_wan.configs import SIZE_CONFIGS, WAN_CONFIGS
        from dreamidv_wan.utils.utils import cache_video

        self.cfg = WAN_CONFIGS[config.model.task]
        self.size = SIZE_CONFIGS[config.model.size]
        if config.model.sample_fps is not None:
            self.cfg.sample_fps = config.model.sample_fps
        self.cache_video = cache_video
        self.pipeline = dreamidv_wan.DreamIDV(
            config=self.cfg,
            checkpoint_dir=config.model.ckpt_dir,
            dreamidv_ckpt=config.model.dreamidv_ckpt,
            device_id=runtime.device,
            rank=runtime.rank,
            t5_fsdp=config.model.t5_fsdp,
            dit_fsdp=config.model.dit_fsdp,
            use_usp=(config.model.ulysses_size > 1 or config.model.ring_size > 1),
            t5_cpu=config.model.t5_cpu,
        )

    def generate_chunk(self, assets: ConditioningAssets, frame_num: int):
        ref_paths = [assets.ref_video, assets.mask_video, assets.ref_image, assets.pose_video]
        return self.pipeline.generate(
            self.config.pipeline.prompt,
            ref_paths,
            size=self.size,
            frame_num=frame_num,
            shift=self.config.model.sample_shift,
            sample_solver=self.config.model.sample_solver,
            sampling_steps=self.config.model.sample_steps,
            guide_scale_img=self.config.model.sample_guide_scale_img,
            seed=self.config.model.base_seed,
            offload_model=self.config.model.offload_model,
        )


class FasterBackend(BaseBackend):
    needs_pose = False

    def __init__(self, config: InferenceConfig, runtime: RuntimeContext):
        super().__init__(config, runtime)
        import dreamidv_wan_faster
        from dreamidv_wan_faster.configs import SIZE_CONFIGS, WAN_CONFIGS
        from dreamidv_wan_faster.utils.utils import cache_video

        self.cfg = WAN_CONFIGS[config.model.task]
        self.size = SIZE_CONFIGS[config.model.size]
        if config.model.sample_fps is not None:
            self.cfg.sample_fps = config.model.sample_fps
        self.cache_video = cache_video
        self.pipeline = dreamidv_wan_faster.DreamIDV(
            config=self.cfg,
            checkpoint_dir=config.model.ckpt_dir,
            dreamidv_ckpt=config.model.dreamidv_ckpt,
            device_id=runtime.device,
            rank=runtime.rank,
            t5_fsdp=config.model.t5_fsdp,
            dit_fsdp=config.model.dit_fsdp,
            use_usp=(config.model.ulysses_size > 1 or config.model.ring_size > 1),
            t5_cpu=config.model.t5_cpu,
        )

    def generate_chunk(self, assets: ConditioningAssets, frame_num: int):
        ref_paths = [assets.ref_video, assets.mask_video, assets.ref_image]
        return self.pipeline.generate(
            self.config.pipeline.prompt,
            ref_paths,
            size=self.size,
            frame_num=frame_num,
            shift=self.config.model.sample_shift,
            sample_solver=self.config.model.sample_solver,
            sampling_steps=self.config.model.sample_steps,
            guide_scale_img=self.config.model.sample_guide_scale_img,
            seed=self.config.model.base_seed,
            offload_model=self.config.model.offload_model,
        )


def create_backend(config: InferenceConfig, runtime: RuntimeContext) -> BaseBackend:
    if config.pipeline.name == "faster":
        return FasterBackend(config, runtime)
    return StandardBackend(config, runtime)
