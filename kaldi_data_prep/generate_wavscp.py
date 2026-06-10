#!/usr/bin/env python3
"""
Generate wav.scp file from WAV files for Kaldi-style speaker diarization.
Format: <recording-id> <full-path-to-wav>
SAFE VERSION: Only includes recordings that have corresponding RTTM labels.
"""
import os
import argparse
from pathlib import Path

def extract_recording_id(wav_filename):
    """
    Extract recording ID from WAV filename.
    Ensures that 'rec_val_0007.wav' results in the ID 'rec_val_0007'.
    """
    # Remove .wav extension regardless of case
    name = wav_filename.replace('.wav', '').replace('.WAV', '')
    # Take only the filename stem if a path was passed
    name = os.path.basename(name)
    return name

def generate_wavscp_safe(wav_dir, rttm_dir, output_file, use_absolute_path=True):
    """
    Generate Kaldi wav.scp file for samples that have RTTM labels.
    """
    wav_path = Path(wav_dir)
    rttm_path = Path(rttm_dir)
    wav_entries = {}
    
    # Find all WAV files in the directory
    wav_files = sorted([p for p in wav_path.iterdir() if p.suffix.lower() == '.wav'])
    print(f"Found {len(wav_files)} WAV files.\n")
    
    for wav_file in wav_files:
        recording_id = extract_recording_id(wav_file.name)
        
        # SAFETY CHECK: Only include if the corresponding RTTM (.txt) exists
        # Ensure the filename stem matches (e.g., rec_val_0007.wav -> rec_val_0007.txt)
        rttm_file = rttm_path / f"{wav_file.stem}.txt"
        
        if rttm_file.exists():
            final_wav_path = str(wav_file.absolute()) if use_absolute_path else str(wav_file)
            wav_entries[recording_id] = final_wav_path
            print(f"Mapped: {wav_file.name} -> {recording_id}")
        else:
            # This skips audio that lacks ground truth to prevent training crashes
            print(f"Skipping {wav_file.name}: No corresponding RTTM found.")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write the wav.scp file (sorted by recording ID)
    with open(output_file, 'w') as f:
        for rec_id in sorted(wav_entries.keys()):
            f.write(f"{rec_id} {wav_entries[rec_id]}\n")
    
    print(f"\n✓ Generated {len(wav_entries)} wav.scp entries in rec_val format")
    print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Kaldi wav.scp with RTTM-synced IDs')
    parser.add_argument('wav_dir', help='Directory containing WAV files')
    parser.add_argument('rttm_dir', help='Directory containing RTTM/Txt files')
    parser.add_argument('output_file', help='Output wav.scp file path')
    parser.add_argument('--relative-path', action='store_true', help='Use relative paths')
    
    args = parser.parse_args()
    generate_wavscp_safe(args.wav_dir, args.rttm_dir, args.output_file, 
                         use_absolute_path=not args.relative_path)