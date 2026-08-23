from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoValidationError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float
    duration_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class ExtractedFrame:
    sample_index: int
    frame_index: int
    timestamp_seconds: float
    path: Path


@dataclass(frozen=True)
class ExtractedVideo:
    metadata: VideoMetadata
    frames: tuple[ExtractedFrame, ...]
    temporary_directory: Path

    def remove(self) -> None:
        shutil.rmtree(self.temporary_directory, ignore_errors=True)


def _validated_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise VideoValidationError(
            "invalid_video_metadata",
            "O vídeo não possui duração ou taxa de quadros válidas.",
        )
    if width <= 0 or height <= 0:
        raise VideoValidationError(
            "invalid_video_metadata", "O vídeo não possui dimensões válidas."
        )
    duration_seconds = frame_count / fps
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise VideoValidationError(
            "invalid_video_metadata", "Não foi possível determinar a duração do vídeo."
        )
    return VideoMetadata(
        frame_count=frame_count,
        fps=round(fps, 6),
        duration_seconds=round(duration_seconds, 6),
        width=width,
        height=height,
    )


def extract_uniform_frames(
    video_path: Path,
    *,
    max_duration_seconds: float,
    max_frame_pixels: int,
    sample_count: int,
    minimum_decoded_frames: int,
) -> ExtractedVideo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError(
            "invalid_video", "O arquivo não contém um vídeo legível."
        )

    temporary_directory = Path(tempfile.mkdtemp(prefix="ms-peso-frames-"))
    try:
        metadata = _validated_metadata(capture)
        if metadata.duration_seconds > max_duration_seconds:
            raise VideoValidationError(
                "video_too_long",
                (
                    f"O vídeo tem {metadata.duration_seconds:.2f} s; "
                    f"o limite é {max_duration_seconds:.2f} s."
                ),
            )
        if metadata.width * metadata.height > max_frame_pixels:
            raise VideoValidationError(
                "video_resolution_too_large",
                (
                    f"Cada quadro possui {metadata.width}x{metadata.height} pixels; "
                    f"o limite é {max_frame_pixels} pixels."
                ),
            )

        target_count = min(sample_count, metadata.frame_count)
        frame_indices = sorted(
            set(
                int(round(value))
                for value in np.linspace(
                    0, metadata.frame_count - 1, num=target_count
                )
            )
        )
        frames: list[ExtractedFrame] = []
        for sample_index, frame_index in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            decoded, image = capture.read()
            if not decoded or image is None:
                continue
            frame_path = temporary_directory / f"frame-{sample_index:03d}.png"
            if not cv2.imwrite(str(frame_path), image):
                continue
            frames.append(
                ExtractedFrame(
                    sample_index=sample_index,
                    frame_index=frame_index,
                    timestamp_seconds=round(frame_index / metadata.fps, 6),
                    path=frame_path,
                )
            )

        if len(frames) < minimum_decoded_frames:
            raise VideoValidationError(
                "insufficient_decoded_frames",
                (
                    "Não foi possível decodificar quadros suficientes do vídeo: "
                    f"{len(frames)} de pelo menos {minimum_decoded_frames}."
                ),
            )
        return ExtractedVideo(
            metadata=metadata,
            frames=tuple(frames),
            temporary_directory=temporary_directory,
        )
    except BaseException:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    finally:
        capture.release()
