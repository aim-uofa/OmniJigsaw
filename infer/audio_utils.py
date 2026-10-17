import numpy as np

def prepare_audio_input(audio_wav, max_audio_seconds=3600.0, target_sr=16000):
    
    # Limit the input to the configured audio budget.
    max_audio_samples = int(max_audio_seconds * target_sr)

    if audio_wav is None:
        audio_wav = np.zeros(160000, dtype=np.float32)

    if len(audio_wav) > max_audio_samples:
        # Keep a centered audio window within the configured budget.
        start = (len(audio_wav) - max_audio_samples) // 2
        audio_wav = audio_wav[start : start + max_audio_samples]
        
    return [audio_wav]
