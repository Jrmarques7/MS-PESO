from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ms_peso.service.backend import BackendPrediction, PredictionBackend
from ms_peso.service.video_aggregation import aggregate_frame_weights
from ms_peso.service.video_frames import (
    VideoMetadata,
    extract_uniform_frames,
)
from ms_peso.service.video_policy import VideoInferencePolicy
from ms_peso.service.video_selection import (
    AssessedFrame,
    FrameSelectionError,
    select_technical_frames,
    technical_quality_score,
)


class VideoPredictionUseCase(Protocol):
    def predict(self, video_path: Path) -> VideoPredictionResult: ...


class VideoInferenceError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class SelectedFramePrediction:
    assessed: AssessedFrame
    backend_result: BackendPrediction


@dataclass(frozen=True)
class VideoPredictionResult:
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
    frames: tuple[SelectedFramePrediction, ...]
    policy: VideoInferencePolicy


class VideoPredictionService:
    def __init__(
        self, backend: PredictionBackend, policy: VideoInferencePolicy
    ) -> None:
        self.backend = backend
        self.policy = policy

    def predict(self, video_path: Path) -> VideoPredictionResult:
        extracted = extract_uniform_frames(
            video_path,
            max_duration_seconds=self.policy.max_duration_seconds,
            max_frame_pixels=self.policy.max_frame_pixels,
            sample_count=self.policy.sample_count,
            minimum_decoded_frames=self.policy.min_valid_frames,
        )
        try:
            assessed = tuple(
                AssessedFrame(
                    frame=frame,
                    quality=(quality := self.backend.assess(frame.path)),
                    technical_score=technical_quality_score(quality),
                )
                for frame in extracted.frames
            )
            try:
                selection = select_technical_frames(assessed, self.policy)
            except FrameSelectionError as exc:
                raise VideoInferenceError(exc.code, exc.detail) from exc

            frame_predictions: list[SelectedFramePrediction] = []
            for selected in selection.selected:
                backend_result = self.backend.predict(selected.frame.path)
                if backend_result.prediction is None:
                    raise RuntimeError(
                        "Um quadro aprovado foi rejeitado durante a inferência."
                    )
                frame_predictions.append(
                    SelectedFramePrediction(
                        assessed=selected,
                        backend_result=backend_result,
                    )
                )

            weights = [
                item.backend_result.prediction.estimated_weight_kg
                for item in frame_predictions
                if item.backend_result.prediction is not None
            ]
            aggregate = aggregate_frame_weights(
                weights,
                max_frame_spread_kg=self.policy.max_frame_spread_kg,
            )

            return VideoPredictionResult(
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
                frames=tuple(frame_predictions),
                policy=self.policy,
            )
        finally:
            extracted.remove()
