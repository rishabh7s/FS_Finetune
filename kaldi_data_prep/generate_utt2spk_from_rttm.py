#!/usr/bin/env python3
"""
Generate utt2spk from RTTM files for speaker diarization.
Format: <utterance-id> <speaker-id>
Forced to use clean filenames (rec_val_xxxx) to ensure Kaldi-style consistency.
"""
import os
import argparse
from pathlib import Path

def parse_rttm(rttm_file):
    """
    Parse RTTM and extract speaker segments.
    Uses rttm_file.stem to ensure the ID is 'rec_val_0007' even if the 
    internal text says 'rec_val_0007.wav'.
    """
    segments = []
    # Use the filename (e.g., rec_val_0001) as the base for the Utterance ID
    file_id_from_name = rttm_file.stem 
    
    with open(rttm_file, 'r') as f:
        for line in f:
            line = line.strip()
            if 'SPEAKER' in line:
                parts = line.split()
                try:
                    # RTTM index for data extraction
                    idx = parts.index('SPEAKER')
                    speaker_data = parts[idx:]
                    
                    if len(speaker_data) < 8: 
                        continue
                    
                    # Field 3: Start Time, Field 4: Duration, Field 7: Speaker ID
                    start_time = float(speaker_data[3])
                    duration = float(speaker_data[4])
                    speaker_id = speaker_data[7]
                    
                    segments.append({
                        'file_id': file_id_from_name,
                        'start': start_time,
                        'duration': duration,
                        'speaker': speaker_id
                    })
                except (ValueError, IndexError):
                    continue
    return segments

def generate_utt2spk_from_rttm(rttm_dir, output_file):
    utt2spk_lines = []
    rttm_path = Path(rttm_dir)
    rttm_files = sorted([p for p in rttm_path.iterdir() if p.suffix.lower() == '.txt'])

    print(f"Generating utt2spk for {len(rttm_files)} files...")

    for rttm_file in rttm_files:
        segments = parse_rttm(rttm_file)
        for seg in segments:
            # Convert to centiseconds for the Utterance ID (Standard Kaldi/EEND format)
            start_cs = int(round(seg['start'] * 100))
            end_cs = int(round((seg['start'] + seg['duration']) * 100))
            
            # Create Unique Utterance ID: rec_val_0001_00000000_00000500
            utt_id = f"{seg['file_id']}_{start_cs:08d}_{end_cs:08d}"
            
            # Format: <utt_id> <speaker_id>
            utt2spk_lines.append(f"{utt_id} {seg['speaker']}\n")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Sort is required for Kaldi-style data processing
        f.writelines(sorted(utt2spk_lines))
        
    print(f"✓ Saved {len(utt2spk_lines)} utt2spk entries to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate utt2spk from RTTM files')
    parser.add_argument('rttm_dir', help='Directory containing RTTM/Txt files')
    parser.add_argument('output_file', help='Output utt2spk file path')
    args = parser.parse_args()
    generate_utt2spk_from_rttm(args.rttm_dir, args.output_file)