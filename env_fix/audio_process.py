import base64
from io import BytesIO
import numpy as np
import av

SAMPLE_RATE = 16000

def _check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        has_audio = len(audio_streams) > 0
        container.close()
        return has_audio
    except Exception as e:
        print(f"Warning checking audio in video {video_path}: {e}")
        return False

def load_audio_with_av(path_or_file, sr=16000, offset=0.0, duration=None):
    """
    Decode audio with PyAV, resample it to mono, and return float32 samples.
    """
    try:
        container = av.open(path_or_file)
    except Exception as e:
        raise ValueError(f"PyAV failed to open {path_or_file}: {e}")

    # Use the first audio stream when multiple streams are available.
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if not audio_stream:
        container.close()
        # Let the caller handle media without an audio stream.
        return np.array([], dtype=np.float32)

    resampler = av.audio.resampler.AudioResampler(
        format='flt',
        layout='mono',
        rate=sr
    )

    # Seek near the offset; frame timestamps determine the decoded start.
    if offset > 0:
        target_timestamp = int(offset / audio_stream.time_base)
        container.seek(target_timestamp, stream=audio_stream)

    audio_data = []

    try:
        for frame in container.decode(audio_stream):
            # Drop frames decoded before the requested offset.
            if offset > 0 and frame.time < offset:
                continue

            # Stop after the requested time interval.
            if duration is not None and frame.time > (offset + duration):
                break

            resampled_frames = resampler.resample(frame)
            for r_frame in resampled_frames:
                audio_data.append(r_frame.to_ndarray().flatten())
    except Exception as e:
        print(f"Warning during decoding {path_or_file}: {e}")
    finally:
        container.close()

    if not audio_data:
        return np.array([], dtype=np.float32)

    full_audio = np.concatenate(audio_data)

    # Cap the decoded sample count at the requested duration.
    if duration is not None:
        max_samples = int(duration * sr)
        if len(full_audio) > max_samples:
            full_audio = full_audio[:max_samples]

    return full_audio.astype(np.float32)


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    """
    Extract audio inputs and optional video audio tracks with PyAV.
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]

    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                data = None
                audio_start = 0.0
                audio_end = None

                # Handle explicit audio items.
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)

                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            # Preserve preloaded NumPy inputs.
                            audios.append(
                                path[int(SAMPLE_RATE * audio_start) : None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        else:
                            # PyAV handles local files and supported URLs uniformly.
                            if path.startswith("file://"):
                                data = path[len("file://") :]
                            else:
                                data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))

                # Extract video audio tracks when requested.
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)

                        # Probe the media through the same PyAV backend.
                        if path.startswith("file://"):
                            check_path = path[len("file://") :]
                        else:
                            check_path = path

                        if not _check_if_video_has_audio(check_path):
                            # Match upstream failure behavior when audio is required.
                            raise ValueError("Video must has audio track when use_audio_in_video=True")

                        data = check_path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue

                # Decode all supported input types through PyAV.
                if data is not None:
                    audio_waveform = load_audio_with_av(
                        data,
                        sr=SAMPLE_RATE,
                        offset=audio_start,
                        duration=(audio_end - audio_start) if audio_end is not None else None
                    )
                    audios.append(audio_waveform)

    if len(audios) == 0:
        audios = None
    return audios
