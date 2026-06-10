#!/usr/bin/env python3
"""
Generate reco2dur file from WAV files for Kaldi-style speaker diarization
Format: <recording-id> <duration-in-seconds>
Forced to match the rec_val_xxxx filename format.
"""
import os
import argparse
import wave
from pathlib import Path


def extract_recording_id(wav_filename):
    """
    Extract recording ID from WAV filename.
    Handles the case where RTTM might have 'rec_val_0007.wav' 
    by ensuring the ID is always just 'rec_val_0007'.
    """
    # Remove .wav extension regardless of case
    name = wav_filename.replace('.wav', '').replace('.WAV', '')
    
    # If there are paths in the string, take only the final filename
    name = os.path.basename(name)
    
    return name


def get_wav_duration(wav_file):
    """Get duration of WAV file in seconds"""
    try:
        with wave.open(str(wav_file), 'r') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"  Error reading {wav_file.name}: {e}")
        return None


def generate_reco2dur(wav_dir, output_file):
    """
    Generate Kaldi reco2dur file
    """
    reco2dur_entries = {}
    
    # Find all WAV files
    wav_files = sorted([p for p in Path(wav_dir).iterdir() 
                       if p.suffix.lower() == '.wav'])
    
    print(f"Found {len(wav_files)} WAV files.\n")
    
    for wav_file in wav_files:
        # This will now consistently produce 'rec_val_xxxx'
        recording_id = extract_recording_id(wav_file.name)
        duration = get_wav_duration(wav_file)
        
        if duration is not None:
            reco2dur_entries[recording_id] = duration
            print(f"Mapped: {wav_file.name} -> {recording_id} ({duration:.2f}s)")
        else:
            print(f"⚠ Skipping {wav_file.name} (could not read duration)")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write reco2dur file (sorted by recording ID)
    with open(output_file, 'w') as f:
        for rec_id in sorted(reco2dur_entries.keys()):
            f.write(f"{rec_id} {reco2dur_entries[rec_id]:.2f}\n")
    
    print(f"\n✓ Generated {len(reco2dur_entries)} reco2dur entries in rec_val format")
    print(f"✓ Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Kaldi reco2dur file using filename-based IDs'
    )
    parser.add_argument('wav_dir', help='Directory containing WAV files')
    parser.add_argument('output_file', help='Output reco2dur file path')
    
    args = parser.parse_args()
    generate_reco2dur(args.wav_dir, args.output_file)