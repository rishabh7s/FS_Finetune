#!/usr/bin/env python3
"""
Master script to prepare complete Kaldi-style data directories
Calls individual generation scripts to create all required files
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_script(script_name, args):
    """Run a Python script with given arguments"""
    cmd = [sys.executable, script_name] + args
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"✗ Error running {script_name}")
        return False
    return True


def prepare_kaldi_data(rttm_dir, wav_dir, output_dir, is_val=False):
    """
    Prepare complete Kaldi-style data directory
    
    Args:
        rttm_dir: Directory containing RTTM files
        wav_dir: Directory containing WAV files
        output_dir: Output directory (will create if doesn't exist)
        is_val: If True, also generate reco2num_spk (for validation/test sets)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("KALDI-STYLE DATA PREPARATION")
    print("="*70)
    print(f"RTTM Directory  : {rttm_dir}")
    print(f"WAV Directory   : {wav_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Mode            : {'VALIDATION/TEST' if is_val else 'TRAINING'}")
    print("="*70)
    
    # Define output files
    segments_file = os.path.join(output_dir, 'segments')
    utt2spk_file = os.path.join(output_dir, 'utt2spk')
    wav_scp_file = os.path.join(output_dir, 'wav.scp')
    reco2dur_file = os.path.join(output_dir, 'reco2dur')
    reco2num_spk_file = os.path.join(output_dir, 'reco2num_spk')
    
    success = True
    
    # 1. Generate segments
    print("\n[1/5] Generating segments file...")
    if not run_script('generate_segments_from_rttm.py', 
                     [rttm_dir, segments_file]):
        success = False
    
    # 2. Generate utt2spk
    print("\n[2/5] Generating utt2spk file...")
    if not run_script('generate_utt2spk_from_rttm.py', 
                     [rttm_dir, utt2spk_file]):
        success = False
    
    # 3. Generate wav.scp
    print("\n[3/5] Generating wav.scp file...")
    if not run_script('generate_wavscp.py', 
                     [wav_dir, wav_scp_file]):
        success = False
    
    # 4. Generate reco2dur
    print("\n[4/5] Generating reco2dur file...")
    if not run_script('generate_reco2dur.py', 
                     [wav_dir, reco2dur_file]):
        success = False
    
    # 5. Generate reco2num_spk (only for validation/test sets)
    if is_val:
        print("\n[5/5] Generating reco2num_spk file...")
        if not run_script('generate_reco2num_spk.py', 
                         [rttm_dir, reco2num_spk_file]):
            success = False
    else:
        print("\n[5/5] Skipping reco2num_spk (training mode)")
    
    # Summary
    print("\n" + "="*70)
    if success:
        print("✓ ALL FILES GENERATED SUCCESSFULLY")
        print("="*70)
        print(f"\nGenerated files in {output_dir}:")
        
        files_to_check = [
            ('segments', segments_file),
            ('utt2spk', utt2spk_file),
            ('wav.scp', wav_scp_file),
            ('reco2dur', reco2dur_file),
        ]
        
        if is_val:
            files_to_check.append(('reco2num_spk', reco2num_spk_file))
        
        for name, filepath in files_to_check:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                with open(filepath) as f:
                    lines = len(f.readlines())
                print(f"  ✓ {name:15s} ({lines:5d} lines, {size:7d} bytes)")
            else:
                print(f"  ✗ {name:15s} (NOT FOUND)")
                success = False
    else:
        print("✗ SOME FILES FAILED TO GENERATE")
        print("="*70)
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Prepare complete Kaldi-style data directory from RTTM and WAV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare training data
  python prepare_kaldi_data.py rttm/ data/train/ data/train
  
  # Prepare validation data (with reco2num_spk)
  python prepare_kaldi_data.py rttm/ data/val/ data/val --val
  
Directory structure after running:
  data/train/
    ├── segments       (utterance-level time boundaries)
    ├── utt2spk        (utterance to speaker mapping)
    ├── wav.scp        (recording to WAV file mapping)
    └── reco2dur       (recording durations)
  
  data/val/
    ├── segments
    ├── utt2spk
    ├── wav.scp
    ├── reco2dur
    └── reco2num_spk   (number of speakers per recording)
        """
    )
    
    parser.add_argument('rttm_dir', 
                       help='Directory containing RTTM files (e.g., rttm/)')
    parser.add_argument('wav_dir', 
                       help='Directory containing WAV files (e.g., data/train/)')
    parser.add_argument('output_dir', 
                       help='Output directory for Kaldi files (e.g., data/train/)')
    parser.add_argument('--val', action='store_true',
                       help='Validation/test mode (generates reco2num_spk)')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.isdir(args.rttm_dir):
        print(f"Error: RTTM directory not found: {args.rttm_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.wav_dir):
        print(f"Error: WAV directory not found: {args.wav_dir}")
        sys.exit(1)
    
    # Run preparation
    success = prepare_kaldi_data(args.rttm_dir, args.wav_dir, 
                                 args.output_dir, args.val)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()