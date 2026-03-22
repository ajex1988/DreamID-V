from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .backends import barrier, create_backend, initialize_runtime
from .chunking import (
    build_chunks,
    get_video_info,
    largest_valid_chunk_size,
    next_valid_frame_count,
    pad_frames,
    read_video_range,
    write_video,
)
from .config import InferenceConfig, load_config
from .preprocessors import ConditioningAssets, prepare_assets


def run_from_yaml(config_path: str):
    config = load_config(config_path)
    return run_inference(config)


def run_inference(config: InferenceConfig):
    import torch

    runtime = initialize_runtime(config)
    logging.info(f"Running unified pipeline '{config.pipeline.name}'")
    logging.info(f"Config: {config}")

    assets = prepare_assets(config, runtime.rank)
    barrier(runtime)

    backend = create_backend(config, runtime)

    source_info = get_video_info(assets.ref_video)
    mask_info = get_video_info(assets.mask_video)
    pose_info = get_video_info(assets.pose_video) if backend.needs_pose and assets.pose_video else None

    requested_chunk_size = config.requested_chunk_size
    effective_chunk_size = largest_valid_chunk_size(requested_chunk_size)
    chunks = (
        build_chunks(source_info.frame_count, effective_chunk_size)
        if config.chunk.enabled
        else [(0, source_info.frame_count)]
    )

    if runtime.rank == 0:
        logging.info(
            f"Total frames={source_info.frame_count}, requested_chunk_size={requested_chunk_size}, "
            f"effective_chunk_size={effective_chunk_size}, chunks={len(chunks)}"
        )
        if mask_info.frame_count != source_info.frame_count:
            logging.warning(
                f"Mask frame count ({mask_info.frame_count}) != source frame count ({source_info.frame_count}). "
                "Mask chunks will be padded with the last available mask frame."
            )
        if pose_info and pose_info.frame_count != source_info.frame_count:
            logging.warning(
                f"Pose frame count ({pose_info.frame_count}) != source frame count ({source_info.frame_count}). "
                "Pose chunks will be padded with the last available pose frame."
            )

    chunk_dir = Path(config.chunk.chunk_dir).expanduser().resolve() if config.chunk.chunk_dir else (
        config.resolve_temp_dir() / "chunks" / config.pipeline.name
    )
    chunk_outputs = []

    for chunk_idx, (start, end) in enumerate(chunks):
        actual_chunk_len = end - start
        padded_chunk_len = next_valid_frame_count(actual_chunk_len)
        chunk_assets = _materialize_chunk_assets(
            config=config,
            assets=assets,
            chunk_dir=chunk_dir,
            chunk_idx=chunk_idx,
            start=start,
            end=end,
            padded_chunk_len=padded_chunk_len,
            source_info=source_info,
            mask_info=mask_info,
            pose_info=pose_info,
            needs_pose=backend.needs_pose,
            rank=runtime.rank,
        )

        barrier(runtime)
        chunk_video = backend.generate_chunk(chunk_assets, padded_chunk_len)
        if runtime.rank == 0 and chunk_video is not None:
            chunk_outputs.append(chunk_video[:, :actual_chunk_len].cpu())
        barrier(runtime)

    if runtime.rank == 0:
        if not chunk_outputs:
            raise RuntimeError("No chunk outputs were generated.")
        full_video = torch.cat(chunk_outputs, dim=1)
        save_file = _resolve_output_path(config)
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        backend.save(full_video, save_file)
        logging.info(f"Saved output to {save_file}")
        return save_file
    return None


def _materialize_chunk_assets(
    config: InferenceConfig,
    assets: ConditioningAssets,
    chunk_dir: Path,
    chunk_idx: int,
    start: int,
    end: int,
    padded_chunk_len: int,
    source_info,
    mask_info,
    pose_info,
    needs_pose: bool,
    rank: int,
) -> ConditioningAssets:
    video_name = Path(assets.ref_video).stem
    chunk_video_path = chunk_dir / f"{video_name}_chunk_{chunk_idx:04d}.mp4"
    chunk_mask_path = chunk_dir / f"{video_name}_mask_{chunk_idx:04d}.mp4"
    chunk_pose_path = chunk_dir / f"{video_name}_pose_{chunk_idx:04d}.mp4"

    if rank == 0:
        should_write = not (
            config.chunk.reuse_chunk_files
            and chunk_video_path.exists()
            and chunk_mask_path.exists()
            and (not needs_pose or chunk_pose_path.exists())
        )
        if should_write:
            video_frames = pad_frames(read_video_range(assets.ref_video, start, end), padded_chunk_len)
            mask_frames = _read_conditioning_range(
                assets.mask_video,
                start,
                end,
                padded_chunk_len,
                mask_info.frame_count,
            )
            write_video(video_frames, str(chunk_video_path), source_info.fps)
            write_video(mask_frames, str(chunk_mask_path), source_info.fps)
            if needs_pose and assets.pose_video:
                pose_frames = _read_conditioning_range(
                    assets.pose_video,
                    start,
                    end,
                    padded_chunk_len,
                    pose_info.frame_count if pose_info else 0,
                )
                write_video(pose_frames, str(chunk_pose_path), source_info.fps)

    return ConditioningAssets(
        ref_video=str(chunk_video_path),
        ref_image=assets.ref_image,
        mask_video=str(chunk_mask_path),
        pose_video=str(chunk_pose_path) if needs_pose else None,
    )


def _read_conditioning_range(
    path: str,
    start: int,
    end: int,
    padded_chunk_len: int,
    total_frames: int,
):
    if total_frames <= 0:
        raise ValueError(f"Conditioning video has no frames: {path}")
    if start >= total_frames:
        tail_frames = read_video_range(path, total_frames - 1, total_frames)
        return pad_frames(tail_frames, padded_chunk_len)
    frames = read_video_range(path, start, min(end, total_frames))
    return pad_frames(frames, padded_chunk_len)


def _resolve_output_path(config: InferenceConfig) -> str:
    if config.output.save_file:
        return str(Path(config.output.save_file).expanduser().resolve())
    output_dir = Path(config.output.output_dir).expanduser().resolve()
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{config.pipeline.name}_{config.model.task}_{config.model.size}_"
        f"{config.model.ulysses_size}_{config.model.ring_size}_{formatted_time}.mp4"
    )
    return str(output_dir / filename)
