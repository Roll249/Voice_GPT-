"""
Utility script for voice training and preprocessing
"""

import os
import subprocess
from pathlib import Path
import torchaudio
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np


def preprocess_audio(input_path, output_path, target_sr=22050):
    """
    Preprocess audio for voice cloning.
    - Convert to mono
    - Resample to target sample rate
    - Normalize volume
    - Remove silence
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",  # Mono
        "-ar", str(target_sr),  # Sample rate
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB",
        output_path
    ]

    subprocess.run(cmd, capture_output=True)
    print(f"Preprocessed: {input_path} -> {output_path}")
    return output_path


def slice_audio(input_path, output_dir, min_silence_len=500, silence_thresh=-40, min_length=3000, max_length=15000):
    """
    Slice audio into clips based on silence.

    Args:
        input_path: Path to audio file
        output_dir: Directory to save clips
        min_silence_len: Minimum silence length (ms) to split
        silence_thresh: Silence threshold (dB)
        min_length: Minimum clip length (ms)
        max_length: Maximum clip length (ms)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio = AudioSegment.from_file(input_path)

    # Split on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=200  # Keep 200ms silence at edges
    )

    # Filter and save
    saved_clips = []
    for i, chunk in enumerate(chunks):
        # Skip too short or too long
        if len(chunk) < min_length or len(chunk) > max_length:
            continue

        output_path = output_dir / f"clip_{i:03d}.wav"
        chunk.export(output_path, format="wav")
        saved_clips.append(output_path)
        print(f"Saved: {output_path} ({len(chunk)/1000:.1f}s)")

    print(f"Total clips: {len(saved_clips)}")
    return saved_clips


def combine_references(audio_paths, output_path, max_duration=60):
    """
    Combine multiple audio files into one reference.

    Args:
        audio_paths: List of paths to audio files
        output_path: Output file path
        max_duration: Maximum duration (seconds)
    """
    all_audio = []
    total_duration = 0
    target_sr = 22050

    for path in audio_paths:
        audio, sr = torchaudio.load(path)

        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        duration = audio.shape[1] / target_sr

        if total_duration + duration > max_duration:
            # Trim to fit
            remaining = max_duration - total_duration
            samples_to_take = int(remaining * target_sr)
            audio = audio[:, :samples_to_take]
            all_audio.append(audio)
            break

        all_audio.append(audio)
        total_duration += duration

    # Concatenate
    combined = torch.cat(all_audio, dim=1)

    # Save
    torchaudio.save(output_path, combined, target_sr)
    print(f"Combined {len(all_audio)} files -> {output_path} ({combined.shape[1]/target_sr:.1f}s)")
    return output_path


def batch_preprocess(input_dir, output_dir, target_sr=22050):
    """Preprocess all audio files in directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

    for file in input_dir.iterdir():
        if file.suffix.lower() in audio_extensions:
            output_path = output_dir / f"{file.stem}_processed.wav"
            preprocess_audio(str(file), str(output_path), target_sr)


if __name__ == "__main__":
    import torch  # Added missing import
    
    print("Voice Training Utilities")
    print("1. Preprocess single audio file")
    print("2. Slice audio into clips")
    print("3. Batch preprocess directory")
    
    # Example usage
    # preprocess_audio("raw_audio.mp3", "processed_audio.wav")
    # slice_audio("long_audio.mp3", "clips/")
    # batch_preprocess("raw_audios/", "processed_audios/")