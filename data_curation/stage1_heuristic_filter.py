#!/usr/bin/env python3
"""Stage-1 signal-based heuristic filtering for OmniJigsaw data curation."""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import ffmpeg
import librosa
import numpy as np
import torch
from tqdm import tqdm


LOGGER = logging.getLogger("omnijigsaw.data_curation.stage1")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi"}


@dataclass(frozen=True)
class FilterConfig:
    """Thresholds for the signal-based filtering stage."""

    max_duration_sec: float = 200.0
    target_sample_rate: int = 16_000
    rms_threshold_db: float = -40.0
    max_silence_ratio: float = 0.70
    min_flux_variance: float = 0.5
    vad_min_ratio: float = 0.30
    vad_max_ratio: float = 0.80
    visual_sample_interval: float = 1.0
    visual_diff_threshold: float = 5.0
    max_static_ratio: float = 0.70


@dataclass(frozen=True)
class VideoInfo:
    """Container and stream metadata required by the filter."""

    duration: float
    video_codec: str
    audio_codec: str


class HeuristicVideoFilter:
    """Apply modal-integrity, visual-dynamics, and acoustic-activity checks."""

    def __init__(self, config: FilterConfig, device: str, silero_repo: str | None) -> None:
        self.config = config
        self.device = self._resolve_device(device)
        self.vad_model, self.get_speech_timestamps = self._load_vad(silero_repo)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Silero VAD but is not available.")
        return torch.device(device)

    def _load_vad(self, silero_repo: str | None):
        if silero_repo:
            repo_path = Path(silero_repo).expanduser().resolve()
            if not repo_path.is_dir():
                raise FileNotFoundError(f"Silero VAD repository not found: {repo_path}")
            repo_or_dir = str(repo_path)
            source = "local"
        else:
            repo_or_dir = "snakers4/silero-vad"
            source = "github"

        LOGGER.info("Loading Silero VAD from %s on %s", repo_or_dir, self.device)
        model, utils = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model="silero_vad",
            source=source,
            force_reload=False,
            onnx=False,
        )
        get_speech_timestamps, _, _, _, _ = utils
        model.eval()
        model.to(self.device)
        return model, get_speech_timestamps

    @staticmethod
    def probe_video(video_path: Path) -> VideoInfo:
        """Read duration and stream codecs with ffprobe."""
        try:
            probe = ffmpeg.probe(str(video_path))
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
            raise RuntimeError(f"ffprobe failed: {stderr.strip()}") from exc

        video_stream = next(
            (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"),
            None,
        )
        audio_stream = next(
            (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"),
            None,
        )
        if video_stream is None or audio_stream is None:
            raise ValueError("Both video and audio streams are required.")

        duration_value = probe.get("format", {}).get("duration") or video_stream.get("duration")
        if duration_value is None:
            raise ValueError("Video duration is unavailable.")
        duration = float(duration_value)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid video duration: {duration}")

        return VideoInfo(
            duration=duration,
            video_codec=str(video_stream.get("codec_name", "")),
            audio_codec=str(audio_stream.get("codec_name", "")),
        )

    def analyze_visual_dynamics(self, video_path: Path) -> tuple[bool, str]:
        """Reject videos dominated by visually static frame transitions."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return False, "video_decode_failed"

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not np.isfinite(fps) or fps <= 0 or total_frames <= 0:
                return False, "invalid_video_timing"

            interval_frames = max(1, int(fps * self.config.visual_sample_interval))
            previous_frame = None
            static_count = 0
            transition_count = 0

            for frame_index in range(0, total_frames, interval_frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                decoded, frame = capture.read()
                if not decoded:
                    break

                gray_frame = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
                if previous_frame is not None:
                    transition_count += 1
                    mean_absolute_difference = float(np.mean(cv2.absdiff(previous_frame, gray_frame)))
                    if mean_absolute_difference < self.config.visual_diff_threshold:
                        static_count += 1
                previous_frame = gray_frame

            if transition_count == 0:
                return True, "visual_check_too_short"

            static_ratio = static_count / transition_count
            if static_ratio > self.config.max_static_ratio:
                return False, f"visually_static:{static_ratio:.6f}"
            return True, f"visual_pass:{static_ratio:.6f}"
        finally:
            capture.release()

    def analyze_audio(self, audio_path: Path, video_duration: float) -> tuple[bool, str]:
        """Reject silent, monotone, or speech-dominated acoustic tracks."""
        try:
            waveform, sample_rate = librosa.load(str(audio_path), sr=self.config.target_sample_rate, mono=True)
            if waveform.size == 0 or float(np.max(np.abs(waveform))) == 0.0:
                return False, "empty_or_silent_audio"

            rms = librosa.feature.rms(y=waveform)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            silence_ratio = float(np.mean(rms_db < self.config.rms_threshold_db))
            if silence_ratio > self.config.max_silence_ratio:
                return False, f"too_silent:{silence_ratio:.6f}"

            onset_envelope = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
            flux_variance = float(np.var(onset_envelope))
            if flux_variance < self.config.min_flux_variance:
                return False, f"low_audio_dynamics:{flux_variance:.6f}"

            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
            speech_timestamps = self.get_speech_timestamps(
                waveform_tensor,
                self.vad_model,
                sampling_rate=self.config.target_sample_rate,
            )
            speech_samples = sum(item["end"] - item["start"] for item in speech_timestamps)
            speech_ratio = float(speech_samples / self.config.target_sample_rate / video_duration)
            if not self.config.vad_min_ratio <= speech_ratio <= self.config.vad_max_ratio:
                return False, f"invalid_speech_ratio:{speech_ratio:.6f}"

            return True, (
                f"audio_pass:silence={silence_ratio:.6f},"
                f"flux={flux_variance:.6f},speech={speech_ratio:.6f}"
            )
        except Exception as exc:
            return False, f"audio_analysis_failed:{exc}"

    def evaluate(self, video_path: Path) -> tuple[bool, str, VideoInfo | None]:
        """Run all stage-1 checks for one video."""
        try:
            info = self.probe_video(video_path)
        except Exception as exc:
            return False, f"modal_integrity_failed:{exc}", None

        if info.duration > self.config.max_duration_sec:
            return False, f"duration_exceeded:{info.duration:.6f}", info

        visual_passed, visual_reason = self.analyze_visual_dynamics(video_path)
        if not visual_passed:
            return False, visual_reason, info

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            audio_path = Path(handle.name)

        try:
            try:
                (
                    ffmpeg.input(str(video_path))
                    .output(
                        str(audio_path),
                        ac=1,
                        ar=self.config.target_sample_rate,
                        loglevel="quiet",
                    )
                    .run(overwrite_output=True)
                )
            except ffmpeg.Error as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
                return False, f"audio_extraction_failed:{stderr.strip()}", info

            audio_passed, audio_reason = self.analyze_audio(audio_path, info.duration)
            if not audio_passed:
                return False, audio_reason, info
        finally:
            audio_path.unlink(missing_ok=True)

        return True, f"passed:{visual_reason};{audio_reason}", info

    @staticmethod
    def standardize_video(source_path: Path, destination_path: Path, info: VideoInfo) -> None:
        """Write an MP4 file with H.264 video and AAC audio."""
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            source_path.suffix.lower() == ".mp4"
            and info.video_codec == "h264"
            and info.audio_codec == "aac"
        ):
            shutil.copy2(source_path, destination_path)
            return

        video_codec = "copy" if info.video_codec == "h264" else "libx264"
        try:
            (
                ffmpeg.input(str(source_path))
                .output(
                    str(destination_path),
                    vcodec=video_codec,
                    acodec="aac",
                    loglevel="error",
                )
                .run(overwrite_output=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
            raise RuntimeError(f"video standardization failed: {stderr.strip()}") from exc


def discover_videos(input_dir: Path, output_dir: Path) -> list[Path]:
    """Discover supported videos while excluding a nested output directory."""
    output_dir = output_dir.resolve()
    videos = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        resolved_path = path.resolve()
        if resolved_path == output_dir or output_dir in resolved_path.parents:
            continue
        videos.append(path)
    return sorted(videos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply OmniJigsaw stage-1 signal-based heuristic filtering to a video directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw videos.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for retained standardized videos.")
    parser.add_argument("--silero-repo", type=str, default=None, help="Optional local Silero VAD repository path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-duration-sec", type=float, default=200.0)
    parser.add_argument("--target-sample-rate", type=int, default=16_000)
    parser.add_argument("--rms-threshold-db", type=float, default=-40.0)
    parser.add_argument("--max-silence-ratio", type=float, default=0.70)
    parser.add_argument("--min-flux-variance", type=float, default=0.5)
    parser.add_argument("--vad-min-ratio", type=float, default=0.30)
    parser.add_argument("--vad-max-ratio", type=float, default=0.80)
    parser.add_argument("--visual-sample-interval", type=float, default=1.0)
    parser.add_argument("--visual-diff-threshold", type=float, default=5.0)
    parser.add_argument("--max-static-ratio", type=float, default=0.70)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Input and output directories must be different.")
    if args.max_duration_sec <= 0 or args.target_sample_rate <= 0 or args.visual_sample_interval <= 0:
        raise ValueError("Duration, sample rate, and visual sampling interval must be positive.")
    for name in ("max_silence_ratio", "vad_min_ratio", "vad_max_ratio", "max_static_ratio"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1], got {value}.")
    if args.vad_min_ratio > args.vad_max_ratio:
        raise ValueError("vad_min_ratio cannot exceed vad_max_ratio.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FilterConfig(
        max_duration_sec=args.max_duration_sec,
        target_sample_rate=args.target_sample_rate,
        rms_threshold_db=args.rms_threshold_db,
        max_silence_ratio=args.max_silence_ratio,
        min_flux_variance=args.min_flux_variance,
        vad_min_ratio=args.vad_min_ratio,
        vad_max_ratio=args.vad_max_ratio,
        visual_sample_interval=args.visual_sample_interval,
        visual_diff_threshold=args.visual_diff_threshold,
        max_static_ratio=args.max_static_ratio,
    )
    video_filter = HeuristicVideoFilter(config, args.device, args.silero_repo)
    video_paths = discover_videos(input_dir, output_dir)
    LOGGER.info("Discovered %d videos under %s", len(video_paths), input_dir)

    stats = Counter()
    for source_path in tqdm(video_paths, desc="Stage-1 filtering"):
        passed, reason, info = video_filter.evaluate(source_path)
        if not passed or info is None:
            stats["rejected"] += 1
            LOGGER.debug("Rejected %s: %s", source_path, reason)
            continue

        relative_path = source_path.relative_to(input_dir).with_suffix(".mp4")
        destination_path = output_dir / relative_path
        try:
            video_filter.standardize_video(source_path, destination_path, info)
            stats["retained"] += 1
        except Exception as exc:
            stats["errors"] += 1
            LOGGER.error("Failed to save %s: %s", source_path, exc)

    LOGGER.info(
        "Stage-1 filtering completed: retained=%d, rejected=%d, errors=%d",
        stats["retained"],
        stats["rejected"],
        stats["errors"],
    )


if __name__ == "__main__":
    main()
