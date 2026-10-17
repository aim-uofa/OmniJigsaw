import os
import random
import torch
import numpy as np
import decord
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import math
import av


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


def load_video_and_audio_data(video_path, audio_path=None, target_sr=16000):
    """Open a lazy video reader and load mono audio, using silence when unavailable."""

    # VideoReader performs lazy frame decoding.
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    vr = decord.VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    
    audio_wav = None
    try:
        if audio_path and os.path.exists(audio_path):
            audio_wav = load_audio_with_av(audio_path, sr=target_sr)
        if audio_wav is None:
            audio_wav = load_audio_with_av(video_path, sr=target_sr)
    except Exception as e:
        print(f"[Warning] Error loading audio for {video_path}: {e}. Using silence.")
        audio_wav = None

    if audio_wav is None:
        duration_sec = total_frames / video_fps
        audio_wav = np.zeros(int(duration_sec * target_sr), dtype=np.float32)

    return vr, audio_wav, video_fps


def extract_audio_by_frames(audio_wav, clip_total_frames, clip_frame_indices, video_fps, target_sr=16000, clip_max_audio_samples=None, clip_audio_boundary=None):
    """
    Extract audio aligned with sampled video frames.
    
    Args:
        audio_wav (np.ndarray): Complete audio waveform.
        clip_frame_indices (list[int]): Absolute sampled-frame indices.
        video_fps (float): Source video frame rate.
        target_sr (int): Target audio sample rate.
        clip_max_audio_samples (int): Maximum number of output audio samples.
        clip_audio_boundary (tuple[int, int], optional): Physical audio bounds.
    
    Returns:
        np.ndarray: Extracted audio waveform.
    """

    final_audio_segment = None

    if audio_wav is not None:
        clip_start, clip_end = clip_audio_boundary if clip_audio_boundary else (0, len(audio_wav))
        
        use_frame_anchored = False
        
        if len(clip_frame_indices) > 1 and clip_max_audio_samples is not None:
            first_frame_time = clip_frame_indices[0] / video_fps
            last_frame_time = clip_frame_indices[-1] / video_fps
            frame_span_seconds = last_frame_time - first_frame_time
            
            clip_max_audio_seconds = clip_max_audio_samples / target_sr
            
            if frame_span_seconds > clip_max_audio_seconds:
                use_frame_anchored = True

        if use_frame_anchored:
            # Frame-anchored extraction allocates an equal budget per frame.
            audio_segments = []
            samples_per_frame = clip_max_audio_samples // len(clip_frame_indices)
            half_samples = samples_per_frame // 2

            for frame_idx in clip_frame_indices:
                timestamp_sec = frame_idx / video_fps
                center_sample_idx = int(timestamp_sec * target_sr)
                
                start_idx = max(clip_start, center_sample_idx - half_samples)
                end_idx = min(clip_end, center_sample_idx + half_samples)
                
                if start_idx < end_idx:
                    seg = audio_wav[start_idx:end_idx]
                    # Pad short windows to keep frame contributions uniform.
                    if len(seg) < samples_per_frame:
                        pad_len = samples_per_frame - len(seg)
                        seg = np.pad(seg, (0, pad_len), 'constant')
                    audio_segments.append(seg)
                else:
                    audio_segments.append(np.zeros(samples_per_frame, dtype=np.float32))
            
            if audio_segments:
                final_audio_segment = np.concatenate(audio_segments)
                
        else:
            # Center a continuous audio window on the sampled frame interval.
            first_frame_sample = int((clip_frame_indices[0] / video_fps) * target_sr)
            last_frame_sample = int((clip_frame_indices[-1] / video_fps) * target_sr)
            
            center_sample = (first_frame_sample + last_frame_sample) // 2
            
            if clip_max_audio_samples is not None:
                half_window = clip_max_audio_samples // 2
                target_start = center_sample - half_window
                target_end = center_sample + half_window
                target_start = max(clip_start, target_start)
                target_end = min(clip_end, target_end)
            else:
                target_start = first_frame_sample
                target_end = last_frame_sample
            
            final_audio_segment = audio_wav[target_start : target_end]

    if final_audio_segment is None or len(final_audio_segment) == 0:
        expected_samples = clip_max_audio_samples if clip_max_audio_samples else int((clip_total_frames/video_fps)*target_sr)
        expected_samples = max(expected_samples, 1000)
        final_audio_segment = np.zeros(expected_samples, dtype=np.float32)
    return final_audio_segment


def process_video_jigsaw_with_audio(vr, audio_wav, video_fps, video_info, max_pixels=None, fps_max_frames=None, target_sr=16000, max_audio_seconds_per_clip=6.0):
    
    total_frames = len(vr)
    
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
    
    clip_max_audio_samples = int(max_audio_seconds_per_clip * target_sr) if max_audio_seconds_per_clip is not None else None

    for abs_start, abs_end in clip_ranges:
        total_frames_per_clip = abs_end - abs_start

        min_frames = math.ceil(fps_min_frames / FRAME_FACTOR) * FRAME_FACTOR
        max_frames_limit = (min(fps_max_frames, total_frames_per_clip) // FRAME_FACTOR) * FRAME_FACTOR
        
        nframes = total_frames_per_clip / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames_limit)
        nframes = int((nframes // FRAME_FACTOR) * FRAME_FACTOR)
        if nframes <= 0: 
            nframes = FRAME_FACTOR

        rel_idx = torch.linspace(0, total_frames_per_clip - 1, nframes).round().long()
        abs_idx = (rel_idx + abs_start).tolist()

        abs_idx = [min(idx, total_frames - 1) for idx in abs_idx]
        
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

        # Restrict audio extraction to the current clip.
        clip_audio_start_idx = int((abs_start / video_fps) * target_sr)
        clip_audio_end_idx = int((abs_end / video_fps) * target_sr)
        clip_audio_start_idx = max(0, clip_audio_start_idx)
        clip_audio_end_idx = min(len(audio_wav), clip_audio_end_idx)

        final_audio_segment = extract_audio_by_frames(
            audio_wav=audio_wav,
            clip_total_frames=total_frames_per_clip,
            clip_frame_indices=abs_idx,
            video_fps=video_fps,
            target_sr=target_sr,
            clip_max_audio_samples=clip_max_audio_samples,
            clip_audio_boundary=(clip_audio_start_idx, clip_audio_end_idx)
        )
        
        processed_clip_audios.append(final_audio_segment)
        
        sample_fps = nframes / max((abs_end - abs_start), 1e-6) * video_fps
        fps_list.append(sample_fps)

    return processed_clip_videos, processed_clip_audios, fps_list


def prepare_jigsaw_inputs(vr, audio_wav, video_fps, num_clips=6, max_pixels=100352, max_frames=12, max_audio_seconds_per_clip=6.0):
    """
    Build shuffled video/audio clips and their ground-truth ordering.
    """
    video_info = {
        'fps': 2,
        "fps_max_frames": 12,
        "fps_min_frames": 2,
        "max_pixels": 100352,
        "min_pixels": 100352,
        'jigsaw_clip_num': num_clips,
    }

    videos, audios, fps_list = process_video_jigsaw_with_audio(
        vr, 
        audio_wav,
        video_fps,
        video_info, 
        max_pixels=max_pixels, 
        fps_max_frames=max_frames,
        max_audio_seconds_per_clip=max_audio_seconds_per_clip
    )

    clip_indices = list(range(num_clips))
    random.shuffle(clip_indices)

    gt_order = [clip_indices.index(i) + 1 for i in range(num_clips)]
    gt_string = ", ".join(map(str, gt_order))

    shuffled_videos = [videos[i] for i in clip_indices]
    shuffled_audios = [audios[i] for i in clip_indices]

    fps = 1.0
    try:
        if not fps_list:
            raise ValueError("FPS list is empty")
            
        ref_fps = fps_list[0]
        assert all(abs(x - ref_fps) < 1e-5 for x in fps_list), f"FPS values vary: {fps_list}"
        fps = ref_fps
    except (AssertionError, ValueError) as e:
        if fps_list:
            fps = sum(fps_list) / len(fps_list)
            print(f"[Warning] Detected inconsistent FPS in clips: {fps_list}. Using average FPS: {fps:.4f}")
        else:
            fps = 1.0
            print("[Warning] FPS list is empty. Defaulting to 1.0")
    
    shuffled_videos_np = [v.numpy() if isinstance(v, torch.Tensor) else v for v in shuffled_videos]

    return shuffled_videos_np, fps, shuffled_audios, gt_string


def prepare_original_inputs(vr, audio_wav, video_fps, max_pixels=100352, max_frames=200, max_audio_seconds=600.0, target_sr=16000):
    
    total_frames = len(vr)

    actual_frames = min(max_frames, total_frames)
    if actual_frames <= 0:
        actual_frames = 1
    
    # Sample the source video uniformly.
    abs_idx = torch.linspace(0, total_frames - 1, actual_frames).round().long().tolist()
    abs_idx = [min(idx, total_frames - 1) for idx in abs_idx]

    batch = vr.get_batch(abs_idx).asnumpy()

    video_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
    _, _, h, w = video_tensor.shape
    out_h, out_w = smart_resize(h, w, factor=IMAGE_FACTOR, min_pixels=max_pixels, max_pixels=max_pixels)
    video_tensor = transforms.functional.resize(
        video_tensor, [out_h, out_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    ).float()

    max_audio_samples = int(max_audio_seconds * target_sr)

    final_audio = extract_audio_by_frames(
        audio_wav=audio_wav,
        clip_total_frames=total_frames,
        clip_frame_indices=abs_idx,
        video_fps=video_fps,
        target_sr=target_sr,
        clip_max_audio_samples=max_audio_samples
    )

    actual_duration = max(total_frames / video_fps, 1e-6)
    fps = len(abs_idx) / actual_duration

    if isinstance(video_tensor, torch.Tensor):
        video_tensor = video_tensor.numpy()
        
    return [video_tensor], [final_audio], fps


def load_video(video_path):
    """
    Load a video container and its frame rate.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    vr = decord.VideoReader(video_path)
    video_fps = vr.get_avg_fps()

    return vr, video_fps


def load_video_and_audio_data_batch(video_paths, audio_paths=None, target_sr=16000, num_workers=4):
    """
    Load multiple video/audio inputs concurrently.

    Args:
        video_paths: Video paths.
        audio_paths: Optional one-to-one audio paths.
        target_sr: Target audio sample rate.
        num_workers: Number of loader threads.

    Returns:
        results: Loaded tuples or ``None`` for failed inputs.
        errors: Per-input error messages or ``None`` on success.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if audio_paths is None:
        audio_paths = [None] * len(video_paths)

    def load_single(idx):
        video_path = video_paths[idx]
        audio_path = audio_paths[idx]
        try:
            result = load_video_and_audio_data(video_path, audio_path, target_sr)
            return (idx, result, None)
        except Exception as e:
            return (idx, None, str(e))

    results = [None] * len(video_paths)
    errors = [None] * len(video_paths)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_single, i) for i in range(len(video_paths))]
        for future in as_completed(futures):
            idx, result, error = future.result()
            results[idx] = result
            errors[idx] = error

    return results, errors


def load_video_batch(video_paths, num_workers=4):
    """
    Load multiple video containers concurrently without audio.

    Args:
        video_paths: Video paths.
        num_workers: Number of loader threads.

    Returns:
        results: Loaded ``(reader, fps)`` tuples or ``None``.
        errors: Per-input error messages or ``None`` on success.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def load_single(idx):
        video_path = video_paths[idx]
        try:
            result = load_video(video_path)
            return (idx, result, None)
        except Exception as e:
            return (idx, None, str(e))

    results = [None] * len(video_paths)
    errors = [None] * len(video_paths)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_single, i) for i in range(len(video_paths))]
        for future in as_completed(futures):
            idx, result, error = future.result()
            results[idx] = result
            errors[idx] = error

    return results, errors


def load_audio_batch(audio_paths, target_sr=16000, num_workers=4):
    """
    Load multiple audio files concurrently.

    Args:
        audio_paths: Audio paths.
        target_sr: Target audio sample rate.
        num_workers: Number of loader threads.

    Returns:
        results: Audio arrays or ``None`` for failed inputs.
        errors: Per-input error messages or ``None`` on success.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def load_single(idx):
        audio_path = audio_paths[idx]
        try:
            result = load_audio_with_av(audio_path, sr=target_sr)
            return (idx, result, None)
        except Exception as e:
            return (idx, None, str(e))

    results = [None] * len(audio_paths)
    errors = [None] * len(audio_paths)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_single, i) for i in range(len(audio_paths))]
        for future in as_completed(futures):
            idx, result, error = future.result()
            results[idx] = result
            errors[idx] = error

    return results, errors


def prepare_video_input(vr, video_fps, max_pixels=100352, max_frames=200):
    
    total_frames = len(vr)

    actual_frames = min(max_frames, total_frames)
    if actual_frames <= 0:
        actual_frames = 1
    
    # Sample the source video uniformly.
    abs_idx = torch.linspace(0, total_frames - 1, actual_frames).round().long().tolist()
    abs_idx = [min(idx, total_frames - 1) for idx in abs_idx]

    batch = vr.get_batch(abs_idx).asnumpy()

    video_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
    _, _, h, w = video_tensor.shape
    out_h, out_w = smart_resize(h, w, factor=IMAGE_FACTOR, min_pixels=max_pixels, max_pixels=max_pixels)
    video_tensor = transforms.functional.resize(
        video_tensor, [out_h, out_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    ).float()

    actual_duration = max(total_frames / video_fps, 1e-6)
    fps = len(abs_idx) / actual_duration

    if isinstance(video_tensor, torch.Tensor):
        video_tensor = video_tensor.numpy()
        
    return [video_tensor], fps


def save_debug_separate_streams(
    video_clips, 
    audio_clips, 
    order_input, 
    output_base_path, 
    fps=2.0, 
    sample_rate=16000
):
    """
    Concatenate shuffled video and audio clips in the requested order.
    
    Outputs a silent MP4 and a separate WAV file.

    Args:
        video_clips: list of numpy arrays or tensors
        audio_clips: list of numpy arrays
        order_input: One-based order string or sequence, e.g. ``"2, 1, 3"``.
        output_base_path: Output prefix without an extension.
        fps: Output video frame rate.
        sample_rate: Output audio sample rate.
    """

    import re
    from fractions import Fraction

    # Convert the external one-based order to zero-based indices.
    if isinstance(order_input, str):
        cleaned_order = re.sub(r'[^\d,]', '', order_input)
        try:
            indices = [int(i.strip()) - 1 for i in cleaned_order.split(',') if i.strip()]
        except ValueError:
            print(f"[Warning] Could not parse order string: '{order_input}'. Skipping.")
            return
    elif isinstance(order_input, (list, tuple)):
        indices = [i - 1 for i in order_input]
    else:
        print("[Error] Invalid order_input format.")
        return

    num_clips = len(video_clips)
    if not indices or any(i < 0 or i >= num_clips for i in indices):
        print(f"[Warning] Invalid indices: {indices}. Total clips: {num_clips}. Skipping.")
        return

    ordered_videos = [video_clips[i] for i in indices]
    ordered_audios = [audio_clips[i] for i in indices]

    print(f">>> Saving debug files to {output_base_path}.[mp4/wav] (Order: {[i+1 for i in indices]})")

    # Write a silent MP4 stream.
    video_path = f"{output_base_path}.mp4"
    container_v = av.open(video_path, mode='w')
    
    ref_vid = ordered_videos[0]
    if isinstance(ref_vid, torch.Tensor):
        ref_vid = ref_vid.numpy()
        
    if ref_vid.ndim == 4 and ref_vid.shape[1] == 3: # (T, 3, H, W)
        height, width = ref_vid.shape[2], ref_vid.shape[3]
        is_channel_first = True
    else: # (T, H, W, 3)
        height, width = ref_vid.shape[1], ref_vid.shape[2]
        is_channel_first = False

    fps_fraction = Fraction(float(fps)).limit_denominator(100000)
    stream_v = container_v.add_stream('h264', rate=fps_fraction)
    stream_v.width = width
    stream_v.height = height
    stream_v.pix_fmt = 'yuv420p'

    for clip_tensor in ordered_videos:
        if isinstance(clip_tensor, torch.Tensor):
            clip_np = clip_tensor.cpu().numpy()
        else:
            clip_np = clip_tensor

        # Convert from channel-first tensors to RGB frames.
        if is_channel_first:
            clip_np = np.transpose(clip_np, (0, 2, 3, 1))
        
        if clip_np.dtype != np.uint8:
            # Video tensors use the 0-255 pixel range.
            clip_np = clip_np.astype(np.uint8)

        for frame_data in clip_np:
            frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
            for packet in stream_v.encode(frame):
                container_v.mux(packet)

    for packet in stream_v.encode():
        container_v.mux(packet)
    
    container_v.close()

    # Write a separate PCM WAV stream.
    audio_path = f"{output_base_path}.wav"
    container_a = av.open(audio_path, mode='w')
    
    stream_a = container_a.add_stream('pcm_s16le', rate=sample_rate)
    stream_a.layout = 'mono'

    full_audio = np.concatenate(ordered_audios).astype(np.float32)

    # Convert normalized float audio to clipped signed 16-bit PCM.
    full_audio = np.clip(full_audio, -1.0, 1.0)
    full_audio_int16 = (full_audio * 32767).astype(np.int16)

    frame_size = 1024
    idx = 0
    while idx < len(full_audio_int16):
        chunk = full_audio_int16[idx : idx + frame_size]
        
        # PyAV expects planar mono audio with shape ``(1, samples)``.
        chunk_2d = chunk.reshape(1, -1) 
        
        frame = av.AudioFrame.from_ndarray(chunk_2d, format='s16', layout='mono')
        frame.rate = sample_rate
        
        for packet in stream_a.encode(frame):
            container_a.mux(packet)
        idx += frame_size

    for packet in stream_a.encode():
        container_a.mux(packet)
        
    container_a.close()
