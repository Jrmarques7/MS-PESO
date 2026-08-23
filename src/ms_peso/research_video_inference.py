from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ms_peso.image_quality import ImageQualityPolicy, ImageQualityReport
from ms_peso.inference import CattleWeightPredictor
from ms_peso.model_package import ModelDescriptor
from ms_peso.service.video_aggregation import aggregate_frame_weights
from ms_peso.service.video_frames import VideoMetadata, extract_uniform_frames
from ms_peso.service.video_policy import VideoInferencePolicy
from ms_peso.service.video_selection import (
    AssessedFrame,
    select_technical_frames,
    technical_quality_score,
)


class ImagePredictor(Protocol):
    @property
    def device(self) -> object: ...

    def predict_image(self, image_path: Path): ...


@dataclass(frozen=True)
class ResearchVideoFramePrediction:
    frame_index: int
    timestamp_seconds: float
    technical_score: float
    estimated_weight_kg: float
    quality: ImageQualityReport
    exported_path: Path | None


@dataclass(frozen=True)
class ResearchVideoPrediction:
    prediction_status: str
    rejection_code: str | None
    rejection_detail: str | None
    estimated_weight_kg: float | None
    frame_spread_kg: float
    median_absolute_deviation_kg: float
    consensus_status: str
    metadata: VideoMetadata
    sampled_frame_count: int
    eligible_frame_count: int
    frames: tuple[ResearchVideoFramePrediction, ...]
    descriptor: ModelDescriptor
    video_policy: VideoInferencePolicy
    device: str


def _default_predictor_factory(
    descriptor: ModelDescriptor, device: str
) -> ImagePredictor:
    return CattleWeightPredictor.load(descriptor, device=device)


def _export_frame(
    assessed: AssessedFrame, destination: Path, *, order: int
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / (
        f"selected-{order:02d}-frame-{assessed.frame.frame_index:06d}.png"
    )
    shutil.copy2(assessed.frame.path, output_path)
    return output_path.resolve()


class B2ResearchVideoPredictor:
    """Runs the frozen research image model over selected video frames."""

    def __init__(
        self,
        *,
        descriptor: ModelDescriptor,
        quality_policy: ImageQualityPolicy,
        video_policy: VideoInferencePolicy,
        device: str,
        predictor_factory=_default_predictor_factory,
    ) -> None:
        self.descriptor = descriptor
        self.quality_policy = quality_policy
        self.video_policy = video_policy
        self.device = device
        self.predictor_factory = predictor_factory

    def predict(
        self,
        video_path: Path,
        *,
        selected_frames_directory: Path | None = None,
    ) -> ResearchVideoPrediction:
        from ms_peso.image_quality import assess_image_quality

        extracted = extract_uniform_frames(
            video_path,
            max_duration_seconds=self.video_policy.max_duration_seconds,
            max_frame_pixels=self.video_policy.max_frame_pixels,
            sample_count=self.video_policy.sample_count,
            minimum_decoded_frames=self.video_policy.min_valid_frames,
        )
        try:
            assessed = tuple(
                AssessedFrame(
                    frame=frame,
                    quality=(
                        quality := assess_image_quality(frame.path, self.quality_policy)
                    ),
                    technical_score=technical_quality_score(quality),
                )
                for frame in extracted.frames
            )
            selection = select_technical_frames(assessed, self.video_policy)
            predictor = self.predictor_factory(self.descriptor, self.device)
            frame_results: list[ResearchVideoFramePrediction] = []
            weights: list[float] = []
            for order, item in enumerate(selection.selected, start=1):
                prediction = predictor.predict_image(item.frame.path)
                weight = float(prediction.estimated_weight_kg)
                weights.append(weight)
                exported_path = (
                    _export_frame(item, selected_frames_directory, order=order)
                    if selected_frames_directory is not None
                    else None
                )
                frame_results.append(
                    ResearchVideoFramePrediction(
                        frame_index=item.frame.frame_index,
                        timestamp_seconds=item.frame.timestamp_seconds,
                        technical_score=item.technical_score,
                        estimated_weight_kg=weight,
                        quality=item.quality,
                        exported_path=exported_path,
                    )
                )
            aggregate = aggregate_frame_weights(
                weights,
                max_frame_spread_kg=self.video_policy.max_frame_spread_kg,
            )
            return ResearchVideoPrediction(
                prediction_status=aggregate.prediction_status,
                rejection_code=aggregate.rejection_code,
                rejection_detail=aggregate.rejection_detail,
                estimated_weight_kg=aggregate.estimated_weight_kg,
                frame_spread_kg=aggregate.frame_spread_kg,
                median_absolute_deviation_kg=(
                    aggregate.median_absolute_deviation_kg
                ),
                consensus_status=aggregate.consensus_status,
                metadata=extracted.metadata,
                sampled_frame_count=selection.assessed_count,
                eligible_frame_count=selection.eligible_count,
                frames=tuple(frame_results),
                descriptor=self.descriptor,
                video_policy=self.video_policy,
                device=str(predictor.device),
            )
        finally:
            extracted.remove()
