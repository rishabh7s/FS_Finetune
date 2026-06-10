#!/usr/bin/env python3
"""
Generate Kaldi-style reco2num_spk file from RTTM/Txt files.
Format: <recording-id> <number-of-speakers>
Forced to use filenames (rec_val_xxxx) to match wav.scp and reco2dur.
"""
import os
import argparse
from pathlib import Path

def parse_rttm(rttm_file):
    """
    Parse RTTM file and extract speakers.
    Uses the FILENAME (stem) as the ID to stay consistent with other scripts.
    """
    speakers = set()
    # Takes 'rec_val_0007.txt' and returns 'rec_val_0007'
    file_id_from_name = rttm_file.stem 
    
    with open(rttm_file, 'r') as f:
        for line in f:
            line = line.strip()
            if 'SPEAKER' in line:
                parts = line.split()
                try:
                    # RTTM index for Speaker ID is typically 7
                    # Format: SPEAKER <file> <chnl> <start> <dur> <conf> <gender> <spkr_id>
                    idx = parts.index('SPEAKER')
                    speaker_data = parts[idx:]
                    
                    if len(speaker_data) < 8:
                        continue
                    
                    # Add unique speaker ID to the set
                    speakers.add(speaker_data[7])
                except (ValueError, IndexError):
                    continue
    return file_id_from_name, speakers

def generate_reco2num_spk_from_rttm(rttm_dir, output_file):
    """
    Groups unique speakers by filename ID and writes count to file.
    """
    reco2num_spk = {}
    
    rttm_path = Path(rttm_dir)
    rttm_files = sorted([p for p in rttm_path.iterdir() if p.suffix.lower() == '.txt'])
    
    if not rttm_files:
        print(f"No .txt files found in {rttm_dir}")
        return

    print(f"Processing {len(rttm_files)} files into rec_val format...\n")
    
    for rttm_file in rttm_files:
        file_id, speakers = parse_rttm(rttm_file)
        
        if not speakers:
            print(f"  Warning: No speakers found in {rttm_file.name}")
            continue
        
        reco2num_spk[file_id] = len(speakers)
        print(f"Mapped: {rttm_file.name} -> {file_id} ({len(speakers)} speakers)")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Write the Kaldi file: <RecordingID> <Count>
    with open(output_file, 'w') as f:
        for rec_id in sorted(reco2num_spk.keys()):
            f.write(f"{rec_id} {reco2num_spk[rec_id]}\n")
    
    print(f"\n✓ Generated {len(reco2num_spk)} entries in rec_val format.")
    print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Kaldi reco2num_spk file')
    parser.add_argument('rttm_dir', help='Directory containing RTTM/Txt files')
    parser.add_argument('output_file', help='Output reco2num_spk file path')
    
    args = parser.parse_args()
    generate_reco2num_spk_from_rttm(args.rttm_dir, args.output_file)