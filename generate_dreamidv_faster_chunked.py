# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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

import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
import random
import torch.distributed as dist
from PIL import Image

import cv2
import numpy as np

import dreamidv_wan_faster
from dreamidv_wan_faster.configs import WAN_CONFIGS, SIZE_CONFIGS
from dreamidv_wan_faster.utils.utils import cache_video, str2bool

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose"))
from pose.extract import process_dwpose


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.dreamidv_ckpt is not None, "Please specify the Phantom-Wan checkpoint."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 12

    if args.sample_shift is None:
        args.sample_shift = 5.0

    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Chunked generation for ultra-long sequences with DreamID-V.")
    parser.add_argument(
        "--task",
        type=str,
        default="swapface",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size (in frames). If not provided, fall back to frame_num.")
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=24,
        help="The fps of the generated video."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--dreamidv_ckpt",
        type=str,
        default=None,
        help="The path to the Phantom-Wan checkpoint.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--ref_image",
        type=str,
        default="./assets/test_case/ref_image/an_1.jpg",
        help="The reference images used by DreamID-V.")
    parser.add_argument(
        "--ref_video",
        type=str,
        default="./assets/test_case/ref_video/a_girl.mp4",
        help="The reference video used by DreamID-V.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale_img",
        type=float,
        default=4.0,
        help="Classifier free guidance scale for reference images.")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def _write_video(frames, path, fps):
    if len(frames) == 0:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[0], frames[0].shape[1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _build_chunks(total_frames, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    chunks = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        # Merge extremely small tail into the previous chunk to avoid tiny segments.
        if end - start < 5 and chunks:
            prev_start, _ = chunks.pop()
            start = prev_start
            end = total_frames
        chunks.append((start, end))
        start = end
    return chunks


def generate_chunked(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), "context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert (
            args.ulysses_size * args.ring_size == world_size
        ), "The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert (
            cfg.num_heads % args.ulysses_size == 0
        ), "`num_heads` must be divisible by `ulysses_size`."

    if args.sample_fps is not None:
        cfg.sample_fps = args.sample_fps

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info("Creating DreamID-V pipeline.")
    wan_swapface = dreamidv_wan_faster.DreamIDV(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        dreamidv_ckpt=args.dreamidv_ckpt,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    prompt = "chang face"
    ref_video_path = args.ref_video
    ref_img_path = args.ref_image
    text_prompt = prompt

    temp_dir = os.path.join(os.path.dirname(ref_video_path), "temp_generated")
    video_name_base = os.path.basename(ref_video_path).split(".")[0]
    final_pose_path = os.path.join(temp_dir, video_name_base + "_pose.mp4")
    final_mask_path = os.path.join(temp_dir, video_name_base + "_mask.mp4")

    if rank == 0:
        if not (os.path.exists(final_pose_path) and os.path.exists(final_mask_path)):
            os.makedirs(temp_dir, exist_ok=True)
            try:
                process_dwpose(ref_video_path, final_pose_path, final_mask_path)
                print(f"[Rank 0] DWPose Success: {final_pose_path}")
            except Exception as dw_e:
                print(f"[Rank 0] DWPose failed: {dw_e}")
        else:
            print(f"[Rank 0] Files exist, skipping DWPose: {final_mask_path}")

    if dist.is_initialized():
        dist.barrier()

    chunk_size = args.chunk_size or args.frame_num
    if chunk_size is None or chunk_size <= 0:
        raise ValueError("chunk_size must be positive. Set --chunk_size or --frame_num.")

    # Prepare video readers to determine chunking.
    from decord import VideoReader

    vr = VideoReader(ref_video_path)
    mask_vr = VideoReader(final_mask_path)
    total_frames = len(vr)
    mask_len = len(mask_vr)
    if mask_len != total_frames and rank == 0:
        logging.warning(
            f"Mask frame count ({mask_len}) != video frame count ({total_frames}). "
            "Padding mask with its last frame to keep lengths aligned.")
    src_fps = max(round(vr.get_avg_fps()), 1)

    if (chunk_size - 1) % 4 != 0 and rank == 0:
        logging.warning(
            f"chunk_size={chunk_size} is not in the form 4n+1; DreamID-V will internally round "
            "each chunk down by up to 3 frames.")

    chunks = _build_chunks(total_frames, chunk_size)
    if rank == 0:
        logging.info(f"Total frames: {total_frames}, chunk_size: {chunk_size}, chunks: {len(chunks)}")

    chunk_outputs = []
    chunk_dir = os.path.join(temp_dir, "chunks")

    for chunk_idx, (start, end) in enumerate(chunks):
        if rank == 0:
            frames = [vr[i].asnumpy() for i in range(start, end)]
            mask_frames = [
                mask_vr[i].asnumpy() if i < mask_len else mask_vr[mask_len - 1].asnumpy()
                for i in range(start, end)
            ]
            # Avoid tiny chunks by padding with last frame.
            while len(frames) < 5:
                frames.append(frames[-1])
                mask_frames.append(mask_frames[-1])

            chunk_video_path = os.path.join(
                chunk_dir, f"{video_name_base}_chunk_{chunk_idx:04d}.mp4")
            chunk_mask_path = os.path.join(
                chunk_dir, f"{video_name_base}_mask_{chunk_idx:04d}.mp4")

            _write_video(frames, chunk_video_path, src_fps)
            _write_video(mask_frames, chunk_mask_path, src_fps)
        else:
            chunk_video_path = os.path.join(
                chunk_dir, f"{video_name_base}_chunk_{chunk_idx:04d}.mp4")
            chunk_mask_path = os.path.join(
                chunk_dir, f"{video_name_base}_mask_{chunk_idx:04d}.mp4")

        if dist.is_initialized():
            dist.barrier()

        ref_paths = [
            chunk_video_path,
            chunk_mask_path,
            ref_img_path,
        ]

        chunk_video = wan_swapface.generate(
            text_prompt,
            ref_paths,
            size=SIZE_CONFIGS[args.size],
            frame_num=chunk_size,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale_img=args.sample_guide_scale_img,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

        if rank == 0 and chunk_video is not None:
            chunk_outputs.append(chunk_video.cpu())

        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        if len(chunk_outputs) == 0:
            raise RuntimeError("No chunk outputs were generated.")
        full_video = torch.cat(chunk_outputs, dim=1)

        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = ".mp4"
            args.save_file = (
                f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_time}"
                + suffix
            )

        cache_video(
            tensor=full_video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"Save file: {args.save_file}")

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate_chunked(args)
