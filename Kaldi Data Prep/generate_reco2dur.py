#!/usr/bin/env python3
"""
Generate reco2dur file from WAV files for Kaldi-style speaker diarization
Format: <recording-id> <duration-in-seconds>
"""
import os
import argparse
import wave
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
    
    Args:
        wav_dir: Directory containing WAV files
        output_file: Output reco2dur file path
    """
    reco2dur_entries = {}
    
    # Find all WAV files
    wav_files = sorted([p for p in Path(wav_dir).iterdir() 
                       if p.suffix.lower() == '.wav'])
    
    print(f"Found {len(wav_files)} WAV files.\n")
    
    for wav_file in wav_files:
        recording_id = extract_recording_id(wav_file.name)
        duration = get_wav_duration(wav_file)
        
        if duration is not None:
            reco2dur_entries[recording_id] = duration
            print(f"{recording_id}: {duration:.2f}s")
        else:
            print(f"⚠ Skipping {wav_file.name} (could not read duration)")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write reco2dur file (sorted by recording ID)
    with open(output_file, 'w') as f:
        for rec_id in sorted(reco2dur_entries.keys()):
            f.write(f"{rec_id} {reco2dur_entries[rec_id]:.2f}\n")
    
    print(f"\n✓ Generated {len(reco2dur_entries)} reco2dur entries")
    print(f"✓ Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Kaldi reco2dur file from WAV files'
    )
    parser.add_argument('wav_dir', help='Directory containing WAV files')
    parser.add_argument('output_file', help='Output reco2dur file path')
    
    args = parser.parse_args()
    generate_reco2dur(args.wav_dir, args.output_file)