#!/usr/bin/env python3
"""
Generate segments file from RTTM files for Kaldi-style speaker diarization.
Format: <utterance-id> <recording-id> <start-time> <end-time>
Forced to use filenames (rec_val_xxxx) as the recording-id.
"""
import os
import argparse
from pathlib import Path

def parse_rttm(rttm_file):
    """
    Parse RTTM and use the FILENAME as the file_id.
    Ensures start and end times are calculated correctly from RTTM fields.
    """
    segments = []
    # Use the filename (e.g., rec_val_0001) as the primary key
    file_id_from_name = rttm_file.stem 
    
    with open(rttm_file, 'r') as f:
        for line in f:
            line = line.strip()
            if 'SPEAKER' in line:
                parts = line.split()
                try:
                    # RTTM Standard: SPEAKER <file> <chnl> <start> <dur> ...
                    idx = parts.index('SPEAKER')
                    speaker_data = parts[idx:]
                    
                    if len(speaker_data) < 5: 
                        continue
                    
                    start_time = float(speaker_data[3])
                    duration = float(speaker_data[4])
                    
                    segments.append({
                        'file_id': file_id_from_name,
                        'start': start_time,
                        'end': start_time + duration
                    })
                except (ValueError, IndexError) as e:
                    print(f"Skipping malformed line in {rttm_file.name}: {line}")
                    continue
    return segments

def generate_segments(rttm_dir, output_file):
    all_segments = []
    rttm_path = Path(rttm_dir)
    # Only look for .txt files
    rttm_files = sorted([p for p in rttm_path.iterdir() if p.suffix.lower() == '.txt'])
    
    print(f"Processing {len(rttm_files)} files into rec_val format...")
    
    for rttm_file in rttm_files:
        segments = parse_rttm(rttm_file)
        for seg in segments:
            # Create centisecond timestamps for the Utterance ID (Kaldi convention)
            start_cs = int(round(seg['start'] * 100))
            end_cs = int(round(seg['end'] * 100))
            
            # Utterance ID format: rec_val_0001_00000000_00000500
            utt_id = f"{seg['file_id']}_{start_cs:08d}_{end_cs:08d}"
            
            all_segments.append({
                'utt_id': utt_id,
                'recording_id': seg['file_id'],
                'start': seg['start'],
                'end': seg['end']
            })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Sort is critical for Kaldi/EEND data loaders
        sorted_segs = sorted(all_segments, key=lambda x: (x['recording_id'], x['start']))
        for seg in sorted_segs:
            f.write(f"{seg['utt_id']} {seg['recording_id']} "
                   f"{seg['start']:.2f} {seg['end']:.2f}\n")
            
    print(f"\n✓ Generated {len(all_segments)} segment entries")
    print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Kaldi segments file from RTTM')
    parser.add_argument('rttm_dir', help='Directory containing RTTM/Txt files')
    parser.add_argument('output_file', help='Output segments file path')
    args = parser.parse_args()
    generate_segments(args.rttm_dir, args.output_file)