import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--OMP_NUM_THREADS', type=int, default=1)
temp_args, _ = parser.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(temp_args.OMP_NUM_THREADS)

import torch
import pytorch_lightning as pl
from nnet.model.streaming_tfm_enc_1dcnn_enc_linear_non_autoreg_pos_enc_l2norm import StreamingTransformerEDADiarization
from nnet.model.onl_tfm_enc_1dcnn_enc_linear_non_autoreg_pos_enc_l2norm import OnlineTransformerDADiarization
from datasets.feature import extract_fbank
from train.utils.make_rttm import make_rttm
import hyperpyyaml
from nnet.utils.copy_params import copy_params_from_masked_to_streaming
from torch.cuda.amp import autocast
import time
import warnings
warnings.filterwarnings("ignore")

# New imports for microphone
import sounddevice as sd
import numpy as np
import queue
import threading
import soundfile as sf
import tempfile
import librosa  # For resampling
from datetime import datetime

class MicrophoneStreamingDiarization:
    """Real-time diarization from microphone using existing FS-EEND streaming model"""
    
    def __init__(self, configs, checkpoint_path, device='cpu', input_sample_rate=None, print_interval=2.0):
        self.configs = configs
        self.device = torch.device(device)
        
        # Real-time printing interval (in seconds)
        self.print_interval = print_interval
        self.last_print_time = 0
        self.session_start_time = None
        
        # Audio settings from config (target sample rate for model)
        self.target_sample_rate = configs["data"]["feat"]["sample_rate"]  # 8000 Hz
        
        # Input sample rate (from microphone)
        # If not specified, use default device sample rate or 44100 Hz
        if input_sample_rate is None:
            try:
                default_device = sd.query_devices(kind='input')
                self.input_sample_rate = int(default_device['default_samplerate'])
                print(f"Using default microphone sample rate: {self.input_sample_rate} Hz")
            except:
                self.input_sample_rate = 44100  # Fallback
                print(f"Using fallback sample rate: {self.input_sample_rate} Hz")
        else:
            self.input_sample_rate = input_sample_rate
            print(f"Using specified input sample rate: {self.input_sample_rate} Hz")
        
        # Check if resampling is needed
        self.needs_resampling = (self.input_sample_rate != self.target_sample_rate)
        if self.needs_resampling:
            print(f"⚠ Resampling enabled: {self.input_sample_rate} Hz → {self.target_sample_rate} Hz")
        else:
            print(f"✓ No resampling needed (both at {self.target_sample_rate} Hz)")
        
        # Chunk settings (based on input sample rate for microphone capture)
        self.chunk_duration = 0.1  # 100ms chunks
        self.chunk_samples = int(self.input_sample_rate * self.chunk_duration)
        
        # Buffers
        self.audio_buffer = []  # Stores RESAMPLED audio at target_sample_rate
        self.audio_queue = queue.Queue()
        
        # Feature extraction params (from config)
        self.context_size = configs["data"]["context_recp"]
        self.feat_type = configs["data"]["feat_type"]
        self.frame_size = configs["data"]["feat"]["win_length"]
        self.frame_shift = configs["data"]["feat"]["hop_length"]
        self.subsampling = configs["data"]["subsampling"]
        
        # Tracking
        self.frame_count = 0
        self.predictions = []
        self.is_running = False
        
        # Real-time speaker tracking
        self.current_speakers = set()  # Active speakers in current window
        self.speaker_history = []  # [(timestamp, speaker_set), ...]
        
        # Initialize models
        print("Loading models...")
        self._initialize_models(checkpoint_path)
        print("✓ Models loaded successfully!")
        
    def _initialize_models(self, checkpoint_path):
        """Initialize masked and streaming models"""
        # Create masked model
        self.masked_model = OnlineTransformerDADiarization(
            n_speakers=self.configs["data"]["num_speakers"],
            in_size=(2 * self.configs["data"]["context_recp"] + 1) * self.configs["data"]["feat"]["n_mels"], 
            **self.configs["model"]["params"],
        ).to(self.device)
        
        # Create streaming model
        self.streaming_model = StreamingTransformerEDADiarization(
            in_size=(2 * self.configs["data"]["context_recp"] + 1) * self.configs["data"]["feat"]["n_mels"], 
            **self.configs["model"]["params"],
        ).to(self.device)
        
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[len('model.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        self.masked_model.load_state_dict(new_state_dict)
        self.masked_model.eval()
        
        # Copy parameters to streaming model
        copy_params_from_masked_to_streaming(self.masked_model, self.streaming_model)
        self.streaming_model.eval()
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Convert to mono
        audio_data = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()
        self.audio_queue.put(audio_data.copy())
    
    def _print_realtime_status(self, current_time):
        """Print real-time diarization status"""
        elapsed = current_time - self.session_start_time
        
        # Convert current speakers set to sorted list for consistent display
        active_speakers = sorted(list(self.current_speakers))
        
        if active_speakers:
            speakers_str = ", ".join([f"Speaker_{spk}" for spk in active_speakers])
            print(f"\n[{elapsed:.1f}s] 🎙️  Active: {speakers_str}")
        else:
            print(f"\n[{elapsed:.1f}s] 🔇 No active speakers")
    
    def process_audio_stream(self):
        """Process audio from queue and run streaming inference"""
        print("🎤 Processing audio stream...")
        
        # Temporary file to accumulate audio for feature extraction
        temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        # Variables for feature extraction
        min_audio_for_feature = self.frame_size  # Minimum samples needed
        accumulated_audio = []
        last_feature_idx = 0
        
        while self.is_running:
            try:
                # Get audio chunk from queue (at input_sample_rate)
                audio_chunk = self.audio_queue.get(timeout=0.5)
                
                # Resample if necessary
                if self.needs_resampling:
                    # Resample from input_sample_rate to target_sample_rate
                    audio_chunk_resampled = librosa.resample(
                        audio_chunk.astype(np.float32),
                        orig_sr=self.input_sample_rate,
                        target_sr=self.target_sample_rate
                    )
                else:
                    audio_chunk_resampled = audio_chunk.astype(np.float32)
                
                # Accumulate RESAMPLED audio
                accumulated_audio.append(audio_chunk_resampled)
                self.audio_buffer.extend(audio_chunk_resampled)
                
                # Check if we have enough audio to extract a new feature
                total_samples = len(self.audio_buffer)
                
                # Calculate how many frames we can extract
                # Each feature frame needs frame_shift samples advancement
                num_frames_possible = (total_samples - self.frame_size) // self.frame_shift + 1
                
                if num_frames_possible > last_feature_idx:
                    # Save current audio buffer to temp file (at target_sample_rate)
                    audio_array = np.array(self.audio_buffer, dtype=np.float32)
                    sf.write(temp_audio_path, audio_array, self.target_sample_rate)
                    
                    # Extract features using existing extract_fbank function
                    feat = extract_fbank(
                        temp_audio_path,
                        context_size=self.context_size,
                        input_transform=self.feat_type,
                        frame_size=self.frame_size,
                        frame_shift=self.frame_shift,
                        subsampling=self.subsampling
                    ).to(self.device)
                    
                    # Process new frames
                    for frame_idx in range(last_feature_idx, feat.shape[0]):
                        feat_t = feat[frame_idx:frame_idx+1].unsqueeze(0)  # (1, 1, D)
                        
                        # Run streaming inference
                        with torch.no_grad():
                            pred_t = self.streaming_model.test(
                                feat_t, 
                                max_nspks=self.configs["data"]["max_speakers"] + 2
                            )
                        
                        if pred_t is not None:
                            self.predictions.append(pred_t)
                            
                            # Apply sigmoid to get probabilities
                            probs = torch.sigmoid(pred_t[0, :, 1:])  # Shape: (1, n_speakers)
                            
                            # Determine active speakers (threshold at 0.5)
                            active = (probs > 0.5).squeeze()
                            active_speaker_ids = torch.where(active)[0].cpu().numpy()
                            
                            # Update current speakers
                            self.current_speakers = set(active_speaker_ids.tolist())
                            
                            self.frame_count += 1
                        
                        # Print real-time status at fixed intervals
                        current_time = time.time()
                        if current_time - self.last_print_time >= self.print_interval:
                            self._print_realtime_status(current_time)
                            self.last_print_time = current_time
                    
                    last_feature_idx = feat.shape[0]
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Cleanup
        try:
            os.unlink(temp_audio_path)
        except:
            pass
    
    def start_streaming(self, duration=None):
        """Start microphone streaming and processing"""
        self.is_running = True
        self.session_start_time = time.time()
        self.last_print_time = self.session_start_time
        
        print("\n" + "="*60)
        print("🎙️  REAL-TIME SPEAKER DIARIZATION")
        print("="*60)
        print(f"Print interval: {self.print_interval}s")
        if duration:
            print(f"Duration: {duration}s")
        else:
            print("Duration: Continuous (Press Ctrl+C to stop)")
        print("="*60 + "\n")
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        process_thread.start()
        
        # Start audio stream
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.input_sample_rate,
                blocksize=self.chunk_samples,
                dtype='float32'
            ):
                print("🔴 Recording started...")
                
                if duration:
                    # Fixed duration
                    time.sleep(duration)
                else:
                    # Continuous until interrupted
                    print("Press Ctrl+C to stop recording...")
                    try:
                        while True:
                            time.sleep(0.1)
                    except KeyboardInterrupt:
                        print("\n⏹️  Stopping recording...")
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping recording...")
        finally:
            self.is_running = False
            process_thread.join(timeout=2.0)
            
            # Process any remaining dummy frames
            print("\n🔄 Processing remaining frames...")
            self._process_dummy_frames()
            
            total_time = time.time() - self.session_start_time
            print(f"✓ Recording finished! Total duration: {total_time:.1f}s")
            print(f"✓ Processed {self.frame_count} frames")
    
    def _process_dummy_frames(self):
        """Process dummy frames to flush the streaming model"""
        if len(self.predictions) == 0:
            return
        
        # Get feature dimension from last prediction
        feat_dim = self.streaming_model.in_size
        conv_delay = self.configs["model"]["params"]["conv_delay"]
        
        for _ in range(conv_delay):
            dummy_feat = torch.zeros(1, 1, feat_dim, device=self.device)
            with torch.no_grad():
                pred_t = self.streaming_model.test(
                    dummy_feat,
                    max_nspks=self.configs["data"]["max_speakers"] + 2,
                    dummy_conv_input=True
                )
            if pred_t is not None:
                self.predictions.append(pred_t)
                self.frame_count += 1
    
    def save_results(self, output_path="realtime_output.rttm"):
        """Save diarization results to RTTM file"""
        if not self.predictions:
            print("⚠️  No predictions to save!")
            return None
        
        print(f"\n💾 Saving results to {output_path}...")
        
        # Concatenate all predictions
        preds = torch.cat(self.predictions, dim=1)
        pred = torch.sigmoid(preds[0][:, 1:])
        pred = pred.detach().cpu().float()
        
        print(f"Final prediction shape: {pred.shape}")
        
        # Generate RTTM using existing make_rttm function
        rttm = make_rttm(
            rec="realtime_recording",
            pred=pred,
            frame_shift=self.configs["data"]["feat"]["hop_length"],
            subsampling=self.configs["data"]["subsampling"],
            sampling_rate=self.configs["data"]["feat"]["sample_rate"]
        )
        
        # Save to file
        print("\n" + "="*60)
        print("📋 FINAL RTTM OUTPUT")
        print("="*60)
        with open(output_path, 'w') as f:
            for spk in rttm:
                for utt in rttm[spk]:
                    f.write(f"{spk} {utt}\n")
                    print(f"{spk} {utt}")
        
        print("="*60)
        print(f"✓ Results saved to: {output_path}")
        return rttm


def predict_from_file(wav_path, configs, test_file, gpus=0):
    """Original file-based prediction (unchanged)"""
    # Set device
    device = torch.device(f"cuda:{gpus}" if torch.cuda.is_available() else "cpu")
    
    # Extract Fbank feature
    feat = extract_fbank(
        wav_path,
        context_size=configs["data"]["context_recp"],
        input_transform=configs["data"]["feat_type"],
        frame_size=configs["data"]["feat"]["win_length"],
        frame_shift=configs["data"]["feat"]["hop_length"],
        subsampling=configs["data"]["subsampling"]
    ).to(device)
    rec = wav_path.split("/")[-1].split(".")[0]
    clip_len = feat.shape[0]

    # Define model
    masked_model = OnlineTransformerDADiarization(
        n_speakers=configs["data"]["num_speakers"],
        in_size=(2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"], 
        **configs["model"]["params"],
    ).to(device)
    
    streaming_model = StreamingTransformerEDADiarization(
        in_size=(2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"], 
        **configs["model"]["params"],
    ).to(device)
    
    # Load ckpt
    state_dict = torch.load(test_file, map_location="cpu", weights_only=False)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[len('model.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    masked_model.load_state_dict(new_state_dict)
    masked_model.eval()
    
    with torch.no_grad():
        masked_pred, _, _ = masked_model.test([feat], [len(feat)], max_nspks=configs["data"]["max_speakers"] + 2)
        masked_pred = torch.sigmoid(masked_pred[0][:, 1:])
    masked_pred = masked_pred.detach().cpu().float()
    print("Masked prediction shape:", masked_pred.shape)
    
    copy_params_from_masked_to_streaming(masked_model, streaming_model)

    # Predict
    preds = []
    st_time = time.time()
    streaming_model.eval()
    with torch.no_grad():
        for t in range(len(feat)):
            feat_t = feat[t:t+1].unsqueeze(0)
            pred_t = streaming_model.test(feat_t, max_nspks=configs["data"]["max_speakers"] + 2)
            if pred_t is not None:
                preds.append(pred_t)
        for _ in range(configs["model"]["params"]["conv_delay"]):
            dummy_feat = torch.zeros(1, 1, feat.shape[-1], device=feat.device)
            pred_t = streaming_model.test(dummy_feat, max_nspks=configs["data"]["max_speakers"] + 2, dummy_conv_input=True)
            if pred_t is not None:
                preds.append(pred_t)
    
    ed_time = time.time()
    print(f"Inference time: {ed_time - st_time:.2f}s")
    
    preds = torch.cat(preds, dim=1)
    pred = torch.sigmoid(preds[0][:, 1:])
    pred = pred.detach().cpu().float()
    print("Streaming prediction shape:", pred.shape)
    
    print("Predictions match:", torch.allclose(pred, masked_pred, atol=1e-4, rtol=1e-4))
    
    # Generate RTTM
    rttm = make_rttm(
        rec=rec, 
        pred=pred, 
        frame_shift=configs["data"]["feat"]["hop_length"],
        subsampling=configs["data"]["subsampling"],
        sampling_rate=configs["data"]["feat"]["sample_rate"]
    )
    return rttm


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Common arguments
    parser.add_argument('--configs', default='./conf/spk_onl_tfm_enc_dec_nonautoreg_infer.yaml', 
                        help='Configuration file path')
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint file path")
    parser.add_argument("--gpus", default=0, type=int, 
                        help="Device id of gpus to use")
    
    # Mode selection
    parser.add_argument("--mode", default="mic", choices=["mic", "file"],
                        help="Mode: 'mic' for microphone, 'file' for WAV file")
    
    # Microphone mode arguments
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration in seconds (None for continuous)")
    parser.add_argument("--output", default="realtime_output.rttm",
                        help="Output RTTM file path")
    parser.add_argument("--input_sample_rate", type=int, default=None,
                        help="Input microphone sample rate (None for auto-detect)")
    parser.add_argument("--print_interval", type=float, default=2.0,
                        help="Interval (in seconds) for printing real-time results")
    
    # File mode arguments (original)
    parser.add_argument("--wav_path", type=str, default=None,
                        help="Path to input wav file (for file mode)")
    parser.add_argument("--test_from_folder", default=None,
                        help="Checkpoint folder (legacy)")
    parser.add_argument("--test_from_file", default=None,
                        help="Checkpoint file (legacy, use --checkpoint instead)")
    
    setup = parser.parse_args()
    
    # Load config
    with open(setup.configs, "r") as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)
    
    # Determine checkpoint path
    checkpoint_path = setup.checkpoint or setup.test_from_file
    if not checkpoint_path:
        print("ERROR: Please provide --checkpoint or --test_from_file")
        exit(1)
    
    # Determine device
    if setup.gpus >= 0 and torch.cuda.is_available():
        device = f"cuda:{setup.gpus}"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    if setup.mode == "mic":
        # ===== MICROPHONE MODE =====
        print("\n🎤 MICROPHONE MODE")
        
        # Create diarizer
        diarizer = MicrophoneStreamingDiarization(
            configs=configs,
            checkpoint_path=checkpoint_path,
            device=device,
            input_sample_rate=setup.input_sample_rate,
            print_interval=setup.print_interval
        )
        
        # Start streaming
        try:
            diarizer.start_streaming(duration=setup.duration)
            
            # Save results
            diarizer.save_results(setup.output)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # ===== FILE MODE (Original) =====
        print("\n📁 FILE MODE")
        
        if not setup.wav_path:
            print("ERROR: Please provide --wav_path for file mode")
            exit(1)
        
        rttm = predict_from_file(
            wav_path=setup.wav_path,
            configs=configs,
            test_file=checkpoint_path,
            gpus=setup.gpus
        )
        
        print("\nResults:")
        for spk in rttm:
            for utt in rttm[spk]:
                print(spk, utt)