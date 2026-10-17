import torch
import numpy as np
import decord
import av
import librosa
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import math
import os
import soundfile as sf
import cv2

FRAME_FACTOR = 2
IMAGE_FACTOR = 28

def split_and_trim_uniform_indices(total_frames, clip_num, trim_ratio=0.1):
    """
    Compute trimmed clip boundaries with a uniform frame count across clips.
    """
    clip_len = total_frames / clip_num
    
    temp_ranges = []
    for i in range(clip_num):
        start = i * clip_len
        end = (i + 1) * clip_len
        
        duration = end - start
        trim_amount = duration * trim_ratio
        
        actual_start = int(start + (trim_amount / 2))
        actual_end = int(end - (trim_amount / 2))
        
        if actual_end <= actual_start:
             actual_end = actual_start + 1
             
        temp_ranges.append((actual_start, actual_end))
    
    lengths = [end - start for start, end in temp_ranges]
    min_len = min(lengths)
    
    final_ranges = []
    for start, _ in temp_ranges:
        new_end = start + min_len
        final_ranges.append((start, new_end))
        
    return final_ranges

def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Scale down toward max_pixels and align dimensions to factor.

    min_pixels is retained for call compatibility and is not applied here.
    """
    scale = math.sqrt(max_pixels / (height * width))
    if scale < 1.0:
        out_h = int(height * scale)
        out_w = int(width * scale)
    else:
        out_h, out_w = height, width
    
    # Align each dimension to the model's spatial factor.
    out_h = (out_h // factor) * factor
    out_w = (out_w // factor) * factor
    return max(factor, out_h), max(factor, out_w)

def load_audio_with_av(path_or_file, sr=16000):
    """
    Decode the first audio stream with PyAV and resample it to mono float32.
    """
    try:
        container = av.open(path_or_file)
    except Exception as e:
        raise ValueError(f"PyAV failed to open {path_or_file}: {e}")

    # Use the first audio stream when the container exposes multiple streams.
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if not audio_stream:
        container.close()
        # Let the caller decide how to handle media without audio.
        return None

    try:
        resampler = av.audio.resampler.AudioResampler(
            format='flt',
            layout='mono',
            rate=sr,
        )
    except Exception as e:
        container.close()
        raise ValueError(f"Failed to initialize AudioResampler: {e}")

    audio_data = []
    
    try:
        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            for r_frame in resampled_frames:
                audio_data.append(r_frame.to_ndarray().flatten())
    except Exception as e:
        print(f"[Warning] Error during PyAV decoding {path_or_file}: {e}")
        # Preserve successfully decoded samples if the stream ends abnormally.
    finally:
        container.close()

    if not audio_data:
        return None

    full_audio = np.concatenate(audio_data)
    return full_audio.astype(np.float32)

def process_video_jigsaw_with_audio(video_info, max_pixels=None, fps_max_frames=None, target_sr=16000, max_audio_seconds=6.0):
    video_path = video_info['path']

    # VideoReader performs lazy frame decoding.
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    
    audio_wav = None
    try:
        audio_wav = load_audio_with_av(video_path, sr=target_sr)
    except Exception as e:
        print(f"[Warning] Error loading audio for {video_path}: {e}. Using silence.")
        audio_wav = None
        
    if audio_wav is None:
        duration_sec = total_frames / video_fps
        audio_wav = np.zeros(int(duration_sec * target_sr), dtype=np.float32)

    jigsaw_clip_num = video_info.get('jigsaw_clip_num', 6)
    fps = video_info.get('fps', 2)
    fps_min_frames = video_info.get('fps_min_frames', 2)
    if fps_max_frames is None:
        fps_max_frames = video_info.get('fps_max_frames', 12)
    min_pixels = video_info.get('min_pixels', 128 * 28 * 28)
    if max_pixels is None:
        max_pixels = video_info.get('max_pixels', 100352)

    clip_ranges = split_and_trim_uniform_indices(total_frames, jigsaw_clip_num, 0.1)

    processed_clip_videos = []
    processed_clip_audios = []
    fps_list = []
    
    max_audio_samples = int(max_audio_seconds * target_sr) if max_audio_seconds is not None else None

    for abs_start, abs_end in clip_ranges:
        total_frames_per_clip = abs_end - abs_start

        min_frames = math.ceil(fps_min_frames / FRAME_FACTOR) * FRAME_FACTOR
        max_frames_limit = (min(fps_max_frames, total_frames_per_clip) // FRAME_FACTOR) * FRAME_FACTOR
        
        nframes = total_frames_per_clip / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames_limit)
        nframes = int((nframes // FRAME_FACTOR) * FRAME_FACTOR)
        if nframes <= 0: nframes = FRAME_FACTOR

        rel_idx = torch.linspace(0, total_frames_per_clip - 1, nframes).round().long()
        abs_idx = (rel_idx + abs_start).tolist()
        
        batch = vr.get_batch(abs_idx).asnumpy()
        clip_video = torch.from_numpy(batch).permute(0, 3, 1, 2)
        
        _, _, h, w = clip_video.shape
        out_h, out_w = smart_resize(h, w, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)
        clip_video = transforms.functional.resize(
            clip_video, [out_h, out_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ).float()
        processed_clip_videos.append(clip_video)

        audio_start_sec = abs_start / video_fps
        audio_end_sec = abs_end / video_fps
        
        audio_start_idx = int(audio_start_sec * target_sr)
        audio_end_idx = int(audio_end_sec * target_sr)
        
        audio_start_idx = max(0, audio_start_idx)
        audio_end_idx = min(len(audio_wav), audio_end_idx)
        
        if audio_end_idx > audio_start_idx:
            clip_audio = audio_wav[audio_start_idx:audio_end_idx]
            if max_audio_samples is not None and len(clip_audio) > max_audio_samples:
                    # Keep a centered audio window within the configured budget.
                    start = (len(clip_audio) - max_audio_samples) // 2
                    clip_audio = clip_audio[start : start + max_audio_samples]
        else:
            expected_samples = int((abs_end - abs_start) / video_fps * target_sr)
            if max_audio_samples is not None:
                expected_samples = min(expected_samples, max_audio_samples)
            expected_samples = max(expected_samples, 1000)
            
            clip_audio = np.zeros(expected_samples, dtype=np.float32)
            
        processed_clip_audios.append(clip_audio)
        
        sample_fps = nframes / max((abs_end - abs_start), 1e-6) * video_fps
        fps_list.append(sample_fps)

    return processed_clip_videos, processed_clip_audios, fps_list


def process_full_video_with_audio(
    video_path: str,
    max_pixels: int | None = None,
    max_frames: int | None = None,
    sample_fps: float = 1.0,
    target_sr: int = 16000,
    min_pixels: int = 128 * 28 * 28,
    add_timestamps: bool = True,
):
    """Load a full video as lightly sampled frames plus its audio track.

    ``add_timestamps=True`` preserves the CMM stage-1 input, while SMS passes
    ``False`` to preserve its original timestamp-free SMS stage-1 input.
    """
    # Decode video frames.
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())
    duration_sec = float(total_frames / video_fps)

    # Decode the corresponding audio track.
    audio_wav = None
    try:
        audio_wav = load_audio_with_av(video_path, sr=target_sr)
    except Exception as e:
        print(f"[Warning] Error loading audio for {video_path}: {e}. Using silence.")
        audio_wav = None

    if audio_wav is None:
        audio_wav = np.zeros(int(duration_sec * target_sr), dtype=np.float32)

    # Sample frames for stage-1 temporal grounding.
    if max_frames is None:
        max_frames = 100
    max_frames = int(max_frames)
    if total_frames <= 0:
        raise ValueError(f"Empty video: {video_path}")

    if sample_fps is None or float(sample_fps) <= 0:
        sample_fps = 1.0
    sample_fps = float(sample_fps)

    # MM:SS timestamps only reflect integer seconds; higher FPS may repeat timestamps, which is acceptable.
    if duration_sec > 0 and max_frames > 0:
        max_fps = float(max_frames) / float(duration_sec)
        sample_fps = min(sample_fps, max_fps)
    sample_fps = max(sample_fps, 1e-6)

    # Sample by target FPS, then hard-cap to max_frames if needed.
    sample_times = np.arange(0.0, duration_sec, 1.0 / sample_fps)
    if sample_times.size == 0:
        sample_times = np.array([0.0], dtype=np.float32)

    abs_idx = []
    for t in sample_times.tolist():
        idx = int(round(t * video_fps))
        idx = max(0, min(total_frames - 1, idx))
        if not abs_idx or idx != abs_idx[-1]:
            abs_idx.append(idx)

    if not abs_idx:
        abs_idx = [0]

    if max_frames > 0 and len(abs_idx) > max_frames:
        sel = torch.linspace(0, len(abs_idx) - 1, max_frames).round().long().tolist()
        abs_idx = [abs_idx[i] for i in sel]

    # Ensure frame count is a multiple of FRAME_FACTOR by trimming tail frames.
    remainder = len(abs_idx) % FRAME_FACTOR
    if remainder != 0 and len(abs_idx) - remainder >= FRAME_FACTOR:
        effective_fps = float(len(abs_idx)) / max(duration_sec, 1e-6)
        trim_seconds = float(remainder) / max(effective_fps, 1e-6)
        abs_idx = abs_idx[:-remainder]
        duration_sec = max(0.0, duration_sec - trim_seconds)
        trim_samples = int(duration_sec * target_sr)
        audio_wav = audio_wav[:trim_samples]

    nframes = len(abs_idx)
    batch = vr.get_batch(abs_idx).asnumpy()
    
    
    
    # Draw an MM:SS timestamp on each sampled frame.
    if add_timestamps and len(batch) > 0:
        h, w, _ = batch[0].shape

        # Scale the overlay typography with the frame width.
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, w / 1280.0)
        thickness = max(2, int(font_scale * 2))
        margin = 20
        
        for i, frame_idx in enumerate(abs_idx):
            current_seconds = frame_idx / video_fps
            minutes = int(current_seconds // 60)
            seconds = int(current_seconds % 60)
            time_text = f"{minutes:02d}:{seconds:02d}"
            
            # Decord frames are RGB, so drawing colors use RGB ordering.
            frame = batch[i]

            (text_width, text_height), baseline = cv2.getTextSize(time_text, font, font_scale, thickness)

            # Place the timestamp in the upper-right corner.
            box_x1 = w - text_width - margin - 10
            box_y1 = margin
            box_x2 = w - margin
            box_y2 = margin + text_height + 10 + baseline
            
            # Clamp the overlay to the frame bounds.
            box_x1 = max(0, box_x1)
            box_y1 = max(0, box_y1)
            
            # Draw a white background for legibility.
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)

            # Draw red text using the frame's RGB channel order.
            text_x = box_x1 + 5
            text_y = box_y1 + text_height + 5
            
            cv2.putText(frame, time_text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
            
    video_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)

    # Resize sampled frames to the model's pixel budget.
    if max_pixels is None:
        max_pixels = 100352
    _, _, h, w = video_tensor.shape
    out_h, out_w = smart_resize(h, w, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)
    video_tensor = transforms.functional.resize(
        video_tensor, [out_h, out_w], interpolation=InterpolationMode.BICUBIC, antialias=True
    ).float()

    sample_fps = float(nframes / max(duration_sec, 1e-6))
    return video_tensor, audio_wav.astype(np.float32), sample_fps, duration_sec, video_fps


def process_video_clips_with_audio_by_time_ranges(
    video_path: str,
    clip_time_ranges: list[tuple[float, float]],
    max_pixels: int | None = None,
    fps_max_frames: int | None = None,
    target_sr: int = 16000,
    fps: float = 1.0,
    fps_min_frames: int = 2,
    max_audio_seconds: float | None = 12.0,
    min_pixels: int = 128 * 28 * 28,
):
    if not clip_time_ranges:
        raise ValueError("clip_time_ranges is empty")

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())
    duration_sec = float(total_frames / video_fps)

    # Load full audio once, then slice.
    audio_wav = None
    try:
        audio_wav = load_audio_with_av(video_path, sr=target_sr)
    except Exception as e:
        print(f"[Warning] Error loading audio for {video_path}: {e}. Using silence.")
        audio_wav = None
    if audio_wav is None:
        audio_wav = np.zeros(int(duration_sec * target_sr), dtype=np.float32)

    max_audio_samples = int(max_audio_seconds * target_sr) if max_audio_seconds is not None else None

    if max_pixels is None:
        max_pixels = 100352
    if fps_max_frames is None:
        fps_max_frames = 12

    videos: list[torch.Tensor] = []
    audios: list[np.ndarray] = []
    sample_fps_list: list[float] = []

    for start_sec, end_sec in clip_time_ranges:
        start_sec = float(max(0.0, start_sec))
        end_sec = float(min(duration_sec, end_sec))

        if not (end_sec > start_sec):
            end_sec = min(duration_sec, start_sec + 0.1)
        
        duration_clip_sec = end_sec - start_sec

        min_frames = int(math.ceil(fps_min_frames / FRAME_FACTOR) * FRAME_FACTOR)
        max_frames_limit = int((fps_max_frames // FRAME_FACTOR) * FRAME_FACTOR)
        if max_frames_limit <= 0:
            max_frames_limit = FRAME_FACTOR

        raw_nframes = duration_clip_sec * fps
        if raw_nframes > max_frames_limit:
            max_duration = max_frames_limit / max(fps, 1e-6)
            mid = (start_sec + end_sec) / 2.0
            new_start = mid - max_duration / 2.0
            new_end = mid + max_duration / 2.0
            if new_start < 0.0:
                new_start = 0.0
                new_end = max_duration
            if new_end > duration_sec:
                new_end = duration_sec
                new_start = max(0.0, duration_sec - max_duration)
            start_sec, end_sec = new_start, new_end
            duration_clip_sec = max(0.0, end_sec - start_sec)
            raw_nframes = duration_clip_sec * fps

        if raw_nframes >= FRAME_FACTOR:
            nframes_floor = int(raw_nframes // FRAME_FACTOR) * FRAME_FACTOR
            if nframes_floor != int(raw_nframes):
                new_duration = nframes_floor / max(fps, 1e-6)
                end_sec = min(start_sec + new_duration, duration_sec)
                duration_clip_sec = max(0.0, end_sec - start_sec)
                raw_nframes = duration_clip_sec * fps
        else:
            nframes_floor = 0

        needs_padding = raw_nframes < min_frames
        if needs_padding:
            nframes = int(math.floor(raw_nframes))
            if nframes <= 0:
                nframes = 1
        else:
            nframes = int(raw_nframes)
            if nframes <= 0:
                nframes = FRAME_FACTOR
            nframes = int((nframes // FRAME_FACTOR) * FRAME_FACTOR)
            if nframes <= 0:
                nframes = FRAME_FACTOR
            if nframes > max_frames_limit:
                nframes = max_frames_limit

        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)

        start_frame = max(0, min(total_frames - 1, start_frame))
        end_frame = max(start_frame + 1, min(total_frames, end_frame))

        actual_clip_frames = end_frame - start_frame
        rel_idx = torch.linspace(0, actual_clip_frames - 1, nframes).round().long()
        abs_idx = (rel_idx + start_frame).tolist()

        batch = vr.get_batch(abs_idx).asnumpy()
        clip_video = torch.from_numpy(batch).permute(0, 3, 1, 2)

        _, _, h, w = clip_video.shape
        out_h, out_w = smart_resize(h, w, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)
        clip_video = transforms.functional.resize(
            clip_video, [out_h, out_w], interpolation=InterpolationMode.BICUBIC, antialias=True
        ).float()
        if needs_padding:
            pad_frames = min_frames - nframes
            pad_tensor = clip_video.new_zeros((pad_frames,) + clip_video.shape[1:])
            clip_video = torch.cat([clip_video, pad_tensor], dim=0)

        videos.append(clip_video)

        audio_start_idx = int(start_sec * target_sr)
        audio_end_idx = int(end_sec * target_sr)
        audio_start_idx = max(0, audio_start_idx)
        audio_end_idx = min(len(audio_wav), audio_end_idx)
        
        if audio_end_idx > audio_start_idx:
            clip_audio = audio_wav[audio_start_idx:audio_end_idx]
        else:
            clip_audio = np.zeros(int(duration_clip_sec * target_sr), dtype=np.float32)

        if max_audio_samples is not None and len(clip_audio) > max_audio_samples:
            start = (len(clip_audio) - max_audio_samples) // 2
            clip_audio = clip_audio[start : start + max_audio_samples]

        if needs_padding:
            min_audio_len = int((min_frames / max(fps, 1e-6)) * target_sr)
            if len(clip_audio) < min_audio_len:
                pad_len = min_audio_len - len(clip_audio)
                clip_audio = np.concatenate([clip_audio, np.zeros(pad_len, dtype=np.float32)])

        audios.append(clip_audio.astype(np.float32))
        sample_fps_list.append(float(fps))

    return videos, audios, sample_fps_list, duration_sec, video_fps


def process_video_jigsaw(video_info, max_pixels=None, fps_max_frames=None):
    video_path = video_info['path']

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()

    jigsaw_clip_num = video_info.get('jigsaw_clip_num', 6)
    fps = video_info.get('fps', 2)
    fps_min_frames = video_info.get('fps_min_frames', 2)
    if fps_max_frames is None:
        fps_max_frames = video_info.get('fps_max_frames', 12)
    min_pixels = video_info.get('min_pixels', 128 * 28 * 28)
    if max_pixels is None:
        max_pixels = video_info.get('max_pixels', 100352)

    clip_ranges = split_and_trim_uniform_indices(total_frames, jigsaw_clip_num, 0.1)

    processed_clip_videos = []
    fps_list = []

    for abs_start, abs_end in clip_ranges:
        total_frames_per_clip = abs_end - abs_start

        min_frames = math.ceil(fps_min_frames / FRAME_FACTOR) * FRAME_FACTOR
        max_frames_limit = (min(fps_max_frames, total_frames_per_clip) // FRAME_FACTOR) * FRAME_FACTOR
        
        nframes = total_frames_per_clip / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames_limit)
        nframes = int((nframes // FRAME_FACTOR) * FRAME_FACTOR)
        if nframes <= 0: nframes = FRAME_FACTOR

        rel_idx = torch.linspace(0, total_frames_per_clip - 1, nframes).round().long()
        abs_idx = (rel_idx + abs_start).tolist()
        
        batch = vr.get_batch(abs_idx).asnumpy()
        clip_video = torch.from_numpy(batch).permute(0, 3, 1, 2)
        
        _, _, h, w = clip_video.shape
        out_h, out_w = smart_resize(h, w, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)
        clip_video = transforms.functional.resize(
            clip_video, [out_h, out_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ).float()
        processed_clip_videos.append(clip_video)
        
        sample_fps = nframes / max((abs_end - abs_start), 1e-6) * video_fps
        fps_list.append(sample_fps)

    return processed_clip_videos, fps_list
