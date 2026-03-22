from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


PIPELINE_NAMES = {"manual", "express", "dwpose", "faster"}


@dataclass
class PipelineConfig:
    name: str = "manual"
    prompt: str = "chang face"
    temp_dir: Optional[str] = None


@dataclass
class ModelConfig:
    task: str = "swapface"
    size: str = "1280*720"
    frame_num: int = 81
    sample_fps: int = 24
    ckpt_dir: Optional[str] = None
    dreamidv_ckpt: Optional[str] = None
    offload_model: Optional[bool] = None
    ulysses_size: int = 1
    ring_size: int = 1
    t5_fsdp: bool = False
    t5_cpu: bool = False
    dit_fsdp: bool = False
    sample_solver: str = "unipc"
    sample_steps: int = 12
    sample_shift: float = 5.0
    sample_guide_scale_img: float = 4.0
    base_seed: int = -1


@dataclass
class InputConfig:
    ref_image: Optional[str] = None
    ref_video: Optional[str] = None
    ref_video_facemask: Optional[str] = None
    ref_video_pose: Optional[str] = None


@dataclass
class ChunkConfig:
    enabled: bool = True
    chunk_size: Optional[int] = None
    chunk_dir: Optional[str] = None
    reuse_preprocessed: bool = True
    reuse_chunk_files: bool = False


@dataclass
class OutputConfig:
    save_file: Optional[str] = None
    output_dir: str = "./results_unified"


@dataclass
class InferenceConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inputs: InputConfig = field(default_factory=InputConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceConfig":
        return cls(
            pipeline=PipelineConfig(**data.get("pipeline", {})),
            model=ModelConfig(**data.get("model", {})),
            inputs=InputConfig(**data.get("inputs", {})),
            chunk=ChunkConfig(**data.get("chunk", {})),
            output=OutputConfig(**data.get("output", {})),
        )

    def validate(self) -> None:
        if self.pipeline.name not in PIPELINE_NAMES:
            raise ValueError(
                f"Unsupported pipeline '{self.pipeline.name}'. Expected one of {sorted(PIPELINE_NAMES)}."
            )
        if not self.model.ckpt_dir:
            raise ValueError("model.ckpt_dir is required.")
        if not self.model.dreamidv_ckpt:
            raise ValueError("model.dreamidv_ckpt is required.")
        if not self.inputs.ref_image:
            raise ValueError("inputs.ref_image is required.")
        if not self.inputs.ref_video:
            raise ValueError("inputs.ref_video is required.")
        if self.pipeline.name == "manual":
            if not self.inputs.ref_video_facemask:
                raise ValueError("inputs.ref_video_facemask is required for pipeline=manual.")
            if not self.inputs.ref_video_pose:
                raise ValueError("inputs.ref_video_pose is required for pipeline=manual.")
        if self.model.frame_num < 5:
            raise ValueError("model.frame_num must be at least 5.")
        if self.chunk.chunk_size is not None and self.chunk.chunk_size < 5:
            raise ValueError("chunk.chunk_size must be at least 5.")

    @property
    def requested_chunk_size(self) -> int:
        return self.chunk.chunk_size or self.model.frame_num

    @property
    def ref_video_path(self) -> Path:
        return Path(self.inputs.ref_video).expanduser().resolve()

    @property
    def ref_image_path(self) -> Path:
        return Path(self.inputs.ref_image).expanduser().resolve()

    def resolve_temp_dir(self) -> Path:
        if self.pipeline.temp_dir:
            return Path(self.pipeline.temp_dir).expanduser().resolve()
        return (self.ref_video_path.parent / "temp_generated").resolve()


def load_config(path: str) -> InferenceConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML configs. Install it with `pip install pyyaml`."
        ) from exc

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    config = InferenceConfig.from_dict(data)
    config.validate()
    return config
