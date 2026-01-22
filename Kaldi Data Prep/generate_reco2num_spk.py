#!/usr/bin/env python3
"""
Generate reco2num_spk file from RTTM files for Kaldi-style speaker diarization
Format: <recording-id> <number-of-speakers>
Typically used for validation/test sets
"""
import os
import argparse
from pathlib import Path
from collections import defaultdict


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


def generate_reco2num_spk_from_rttm(rttm_dir, output_file):
    """
    Generate Kaldi reco2num_spk file from RTTM files
    
    Args:
        rttm_dir: Directory containing RTTM files (.txt)
        output_file: Output reco2num_spk file path
    """
    reco2num_spk = defaultdict(set)
    
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
        
        # Count unique speakers for this recording
        for seg in segments:
            reco2num_spk[seg['file_id']].add(seg['speaker'])
        
        num_speakers = len(reco2num_spk[file_id])
        speakers = sorted(reco2num_spk[file_id])
        print(f"  Recording: {file_id}")
        print(f"  Speakers: {num_speakers} ({', '.join(speakers)})")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write reco2num_spk file (sorted by recording ID)
    with open(output_file, 'w') as f:
        for rec_id in sorted(reco2num_spk.keys()):
            num_speakers = len(reco2num_spk[rec_id])
            f.write(f"{rec_id} {num_speakers}\n")
    
    print(f"\n✓ Generated {len(reco2num_spk)} reco2num_spk entries")
    print(f"✓ Saved to: {output_file}")
    
    # Print statistics
    speaker_counts = [len(spks) for spks in reco2num_spk.values()]
    if speaker_counts:
        print(f"\nStatistics:")
        print(f"  Min speakers per recording: {min(speaker_counts)}")
        print(f"  Max speakers per recording: {max(speaker_counts)}")
        print(f"  Avg speakers per recording: {sum(speaker_counts)/len(speaker_counts):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Kaldi reco2num_spk file from RTTM files'
    )
    parser.add_argument('rttm_dir', help='Directory containing RTTM files')
    parser.add_argument('output_file', help='Output reco2num_spk file path')
    
    args = parser.parse_args()
    generate_reco2num_spk_from_rttm(args.rttm_dir, args.output_file)