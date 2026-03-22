from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def build_chunks(total_frames: int, chunk_size: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    chunks: List[Tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        if end - start < 5 and chunks:
            prev_start, _ = chunks.pop()
            start = prev_start
            end = total_frames
        chunks.append((start, end))
        start = end
    return chunks


def largest_valid_chunk_size(max_chunk_size: int) -> int:
    if max_chunk_size < 5:
        raise ValueError("chunk_size must be at least 5 because DreamID-V requires 4n+1 frames.")
    return ((max_chunk_size - 1) // 4) * 4 + 1


def next_valid_frame_count(frame_count: int) -> int:
    if frame_count <= 5:
        return 5
    return ((frame_count - 1 + 3) // 4) * 4 + 1


def get_video_info(path: str) -> VideoInfo:
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return VideoInfo(fps=max(fps, 1.0), width=width, height=height, frame_count=frame_count)


def read_video_range(path: str, start: int, end: int) -> List["np.ndarray"]:
    import cv2

    frames = []
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while index < end:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame[..., ::-1])
            index += 1
    finally:
        capture.release()
    return frames


def pad_frames(frames: Sequence["np.ndarray"], target_length: int) -> List["np.ndarray"]:
    if not frames:
        raise ValueError("Cannot pad an empty frame sequence.")
    padded = [frame.copy() for frame in frames]
    while len(padded) < target_length:
        padded.append(padded[-1].copy())
    return padded


def write_video(frames: Sequence["np.ndarray"], path: str, fps: float) -> None:
    import cv2

    if not frames:
        raise ValueError("No frames to write.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[0], frames[0].shape[1]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame[..., ::-1])
    finally:
        writer.release()
