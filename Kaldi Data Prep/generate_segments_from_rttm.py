#!/usr/bin/env python3
"""
Generate segments file from RTTM files for Kaldi-style speaker diarization
Format: <utterance-id> <recording-id> <start-time> <end-time>
"""
import os
import argparse
from pathlib import Path


def parse_rttm(rttm_file):
    """Parse RTTM file and extract speaker segments"""
    segments = []
    
    with open(rttm_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('SPEAKER'):
                parts = line.split()
                if len(parts) < 8:
                    continue
                
                file_id = parts[1]
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                
                segments.append({
                    'file_id': file_id,
                    'start': start_time,
                    'end': start_time + duration,
                    'speaker': speaker_id
                })
    
    return segments


def generate_segments_from_rttm(rttm_dir, output_file):
    """
    Generate Kaldi segments file from RTTM files
    
    Args:
        rttm_dir: Directory containing RTTM files (.txt)
        output_file: Output segments file path
    """
    all_segments = []
    
    # Find all RTTM files
    rttm_files = sorted([p for p in Path(rttm_dir).iterdir() 
                        if p.suffix.lower() == '.txt' and 'rttm' in p.name.lower()])
    
    print(f"Found {len(rttm_files)} RTTM files.\n")
    
    for rttm_file in rttm_files:
        print(f"Processing {rttm_file.name}...")
        segments = parse_rttm(rttm_file)
        
        if not segments:
            print(f"  Warning: No segments found in {rttm_file.name}")
            continue
        
        file_id = segments[0]['file_id']
        
        # Create segment entries
        for seg in segments:
            # Create utterance ID: recordingID_starttime_endtime
            start_cs = int(seg['start'] * 100)  # centiseconds
            end_cs = int(seg['end'] * 100)
            utt_id = f"{seg['file_id']}_{start_cs:08d}_{end_cs:08d}"
            
            all_segments.append({
                'utt_id': utt_id,
                'recording_id': seg['file_id'],
                'start': seg['start'],
                'end': seg['end']
            })
        
        print(f"  Extracted {len(segments)} segments")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write segments file (sorted by recording_id, then start time)
    with open(output_file, 'w') as f:
        for seg in sorted(all_segments, key=lambda x: (x['recording_id'], x['start'])):
            f.write(f"{seg['utt_id']} {seg['recording_id']} "
                   f"{seg['start']:.2f} {seg['end']:.2f}\n")
    
    print(f"\n✓ Generated {len(all_segments)} segment entries")
    print(f"✓ Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Kaldi segments file from RTTM files'
    )
    parser.add_argument('rttm_dir', help='Directory containing RTTM files')
    parser.add_argument('output_file', help='Output segments file path')
    
    args = parser.parse_args()
    generate_segments_from_rttm(args.rttm_dir, args.output_file)