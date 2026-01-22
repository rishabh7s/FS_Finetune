#!/usr/bin/env python3
"""
Generate utt2spk from RTTM files for speaker diarization
"""
import os
import argparse
from pathlib import Path

def parse_rttm(rttm_file):
    """Parse RTTM file and extract speaker segments"""
    segments = []
    
    with open(rttm_file, 'r') as f:
        for line in f:
            # Use strip() to handle leading/trailing whitespace
            line = line.strip()
            if line.startswith('SPEAKER'):
                parts = line.split()
                if len(parts) < 8: continue
                
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

def generate_utt2spk_from_rttm(rttm_dir, output_file, segment_length=None):
    utt2spk_lines = []
    rttm_path = Path(rttm_dir)
    
    # FIX: Robust file finding (handles .rttm and .RTTM)
    rttm_files = sorted([p for p in Path(rttm_dir).iterdir() if p.suffix.lower() == '.txt'])
    
    print(f"Found {len(rttm_files)} RTTM files.")

    for rttm_file in rttm_files:
        print(f"Processing {rttm_file.name}...")
        segments = parse_rttm(rttm_file)
        
        if not segments:
            print(f"  Warning: No 'SPEAKER' lines found in {rttm_file.name}")
            continue
        
        file_id = segments[0]['file_id']
        
        if segment_length:
            # Create fixed-length segments
            max_time = max(seg['end'] for seg in segments)
            for start in range(0, int(max_time), segment_length):
                end = start + segment_length
                speaker_times = {}
                for seg in segments:
                    if seg['start'] < end and seg['end'] > start:
                        overlap_start = max(seg['start'], start)
                        overlap_end = min(seg['end'], end)
                        duration = overlap_end - overlap_start
                        speaker = seg['speaker']
                        speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
                
                if speaker_times:
                    dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
                    # Format ID for Kaldi/EEND: fileID_start_end
                    utt_id = f"{file_id}_{int(start*100):08d}_{int(end*100):08d}"
                    utt2spk_lines.append(f"{utt_id} {dominant_speaker}\n")
        else:
            # FIX: Instead of just saying "multiple", let's create entries for each unique segment
            # to ensure the file isn't empty and is actually useful for training.
            for seg in segments:
                # Create a unique ID for every segment in the RTTM
                utt_id = f"{file_id}_{int(seg['start']*100):08d}_{int(seg['end']*100):08d}"
                utt2spk_lines.append(f"{utt_id} {seg['speaker']}\n")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.writelines(sorted(utt2spk_lines))
    
    print(f"\nGenerated {len(utt2spk_lines)} utterances")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate utt2spk from RTTM files')
    parser.add_argument('rttm_dir', help='Directory containing RTTM files')
    parser.add_argument('output_file', help='Output utt2spk file path')
    parser.add_argument('--segment-length', type=int, default=None,
                        help='Create fixed-length segments (seconds).')
    
    args = parser.parse_args()
    generate_utt2spk_from_rttm(args.rttm_dir, args.output_file, args.segment_length)