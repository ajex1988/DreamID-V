from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import InferenceConfig


@dataclass
class ConditioningAssets:
    ref_video: str
    ref_image: str
    mask_video: str
    pose_video: Optional[str] = None


def prepare_assets(config: InferenceConfig, rank: int) -> ConditioningAssets:
    pipeline_name = config.pipeline.name
    if pipeline_name == "manual":
        return ConditioningAssets(
            ref_video=str(config.ref_video_path),
            ref_image=str(config.ref_image_path),
            mask_video=str(Path(config.inputs.ref_video_facemask).expanduser().resolve()),
            pose_video=str(Path(config.inputs.ref_video_pose).expanduser().resolve()),
        )
    if pipeline_name == "express":
        return _prepare_express_assets(config, rank)
    if pipeline_name in {"dwpose", "faster"}:
        return _prepare_dwpose_assets(config, rank)
    raise ValueError(f"Unsupported pipeline: {pipeline_name}")


def _prepare_dwpose_assets(config: InferenceConfig, rank: int) -> ConditioningAssets:
    temp_dir = config.resolve_temp_dir()
    video_name_base = config.ref_video_path.stem
    pose_path = temp_dir / f"{video_name_base}_pose.mp4"
    mask_path = temp_dir / f"{video_name_base}_mask.mp4"

    if rank == 0 and not (
        config.chunk.reuse_preprocessed and pose_path.exists() and mask_path.exists()
    ):
        temp_dir.mkdir(parents=True, exist_ok=True)
        from pose.extract import process_dwpose

        process_dwpose(str(config.ref_video_path), str(pose_path), str(mask_path))

    return ConditioningAssets(
        ref_video=str(config.ref_video_path),
        ref_image=str(config.ref_image_path),
        mask_video=str(mask_path),
        pose_video=str(pose_path),
    )


def _prepare_express_assets(config: InferenceConfig, rank: int) -> ConditioningAssets:
    temp_dir = config.resolve_temp_dir()
    video_name_base = config.ref_video_path.stem
    pose_path = temp_dir / f"{video_name_base}_pose.mp4"
    mask_path = temp_dir / f"{video_name_base}_mask.mp4"

    if rank == 0 and not (
        config.chunk.reuse_preprocessed and pose_path.exists() and mask_path.exists()
    ):
        temp_dir.mkdir(parents=True, exist_ok=True)
        _generate_pose_and_mask_videos_express(
            ref_video_path=str(config.ref_video_path),
            ref_image_path=str(config.ref_image_path),
            pose_output_path=str(pose_path),
            mask_output_path=str(mask_path),
        )

    return ConditioningAssets(
        ref_video=str(config.ref_video_path),
        ref_image=str(config.ref_image_path),
        mask_video=str(mask_path),
        pose_video=str(pose_path),
    )


def _generate_pose_and_mask_videos_express(
    ref_video_path: str,
    ref_image_path: str,
    pose_output_path: str,
    mask_output_path: str,
) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    from express_adaption.get_video_npy import get_video_npy
    from express_adaption.media_pipe import FaceMeshAlign_dreamidv, FaceMeshDetector

    detector = FaceMeshDetector()
    align_motion = FaceMeshAlign_dreamidv()

    core_landmark_indices = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        1, 2, 5, 6, 48, 64, 94, 98, 168, 195, 197, 278, 294, 324, 327, 4, 24,
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
        468, 473, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 70, 63, 105, 66, 107,
        300, 293, 334, 296, 336, 156,
    ]
    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]
    core_landmark_indices = list(set(core_landmark_indices + face_oval_indices))

    def save_visualization_video(landmarks_sequence, output_filename, frame_size, fps, mode):
        width, height = frame_size
        writer = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_filename}")
        try:
            for frame_landmarks in landmarks_sequence:
                frame_image = np.zeros((height, width, 3), dtype=np.uint8)
                if mode == "points":
                    for landmark in frame_landmarks:
                        x, y = int(landmark[0]), int(landmark[1])
                        cv2.circle(frame_image, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
                else:
                    face_oval_points = frame_landmarks.astype(np.int32)
                    cv2.fillConvexPoly(frame_image, face_oval_points, color=(255, 255, 255))
                writer.write(frame_image)
        finally:
            writer.release()

    capture = cv2.VideoCapture(ref_video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open reference video: {ref_video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    capture.release()

    face_results = get_video_npy(ref_video_path)
    if not face_results:
        raise RuntimeError("No face landmarks were extracted from the reference video.")

    image = Image.open(ref_image_path).convert("RGB")
    ref_image = np.array(image)
    _, ref_img_lmk = detector(ref_image)
    _, pose_addvis = align_motion(face_results, ref_img_lmk)
    width, height = face_results[0]["width"], face_results[0]["height"]

    core_landmarks_sequence = pose_addvis[:, core_landmark_indices, :]
    save_visualization_video(
        landmarks_sequence=core_landmarks_sequence,
        output_filename=pose_output_path,
        frame_size=(width, height),
        fps=fps,
        mode="points",
    )
    face_oval_sequence = pose_addvis[:, face_oval_indices, :]
    save_visualization_video(
        landmarks_sequence=face_oval_sequence,
        output_filename=mask_output_path,
        frame_size=(width, height),
        fps=fps,
        mode="mask",
    )
