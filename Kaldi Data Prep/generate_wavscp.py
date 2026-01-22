#!/usr/bin/env python3
"""
Generate wav.scp file from WAV files for Kaldi-style speaker diarization
Format: <recording-id> <full-path-to-wav>
"""
import os
import argparse
from pathlib import Path


def extract_recording_id(wav_filename):
    """
    Extract recording ID from WAV filename
    Examples:
        mix_0000176.wav -> data_simu_overlap_split_10w_wav_swb_sre_ts_ns4_beta9_500_36_mix_0000176
    """
    # Remove .wav extension
    name = wav_filename.replace('.wav', '')
    
    # If it's in format mix_NNNN, construct full ID
    if name.startswith('mix_'):
        mix_num = name.split('_')[1]
        # Match the format from RTTM files
        recording_id = f"data_simu_overlap_split_10w_wav_swb_sre_ts_ns4_beta9_500_36_mix_{mix_num}"
        return recording_id
    
    # Otherwise use as-is
    return name


def generate_wavscp(wav_dir, output_file, use_absolute_path=True):
    """
    Generate Kaldi wav.scp file
    
    Args:
        wav_dir: Directory containing WAV files
        output_file: Output wav.scp file path
        use_absolute_path: If True, use absolute paths in wav.scp
    """
    wav_entries = {}
    
    # Find all WAV files
    wav_files = sorted([p for p in Path(wav_dir).iterdir() 
                       if p.suffix.lower() == '.wav'])
    
    print(f"Found {len(wav_files)} WAV files.\n")
    
    for wav_file in wav_files:
        recording_id = extract_recording_id(wav_file.name)
        
        if use_absolute_path:
            wav_path = str(wav_file.absolute())
        else:
            wav_path = str(wav_file)
        
        wav_entries[recording_id] = wav_path
        print(f"{recording_id} -> {wav_file.name}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write wav.scp file (sorted by recording ID)
    with open(output_file, 'w') as f:
        for rec_id in sorted(wav_entries.keys()):
            f.write(f"{rec_id} {wav_entries[rec_id]}\n")
    
    print(f"\n✓ Generated {len(wav_entries)} wav.scp entries")
    print(f"✓ Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Kaldi wav.scp file from WAV files'
    )
    parser.add_argument('wav_dir', help='Directory containing WAV files')
    parser.add_argument('output_file', help='Output wav.scp file path')
    parser.add_argument('--relative-path', action='store_true',
                       help='Use relative paths instead of absolute paths')
    
    args = parser.parse_args()
    generate_wavscp(args.wav_dir, args.output_file, 
                   use_absolute_path=not args.relative_path)