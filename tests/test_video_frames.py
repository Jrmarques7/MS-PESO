from pathlib import Path

import cv2
import numpy as np

from ms_peso.service.video_frames import extract_uniform_frames


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
    )
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((64, 96, 3), index * 4, dtype=np.uint8))
    writer.release()


def test_extracts_uniform_frames_and_cleans_temporary_directory(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path)
    extracted = extract_uniform_frames(
        video_path,
        max_duration_seconds=5,
        max_frame_pixels=96 * 64,
        sample_count=5,
        minimum_decoded_frames=3,
    )
    temporary_directory = extracted.temporary_directory

    assert extracted.metadata.frame_count == 30
    assert extracted.metadata.duration_seconds == 3.0
    assert len(extracted.frames) == 5
    assert extracted.frames[0].frame_index == 0
    assert extracted.frames[-1].frame_index == 29
    assert all(frame.path.is_file() for frame in extracted.frames)

    extracted.remove()
    assert not temporary_directory.exists()
