import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--OMP_NUM_THREADS', type=int, default=1)
temp_args, _ = parser.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(temp_args.OMP_NUM_THREADS)

import random
import torch
import yaml
import hyperpyyaml
import numpy as np
import pytorch_lightning as pl

from functools import partial
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from nnet.model.onl_tfm_enc_1dcnn_enc_linear_non_autoreg_pos_enc_l2norm import OnlineTransformerDADiarization
from utlis.scheduler import NoamScheduler
from datasets.diarization_dataset import KaldiDiarizationDataset, my_collate
from train.oln_tfm_enc_dec import SpeakerDiarization
from nnet.lora.lora_layers import apply_lora_to_model, get_lora_parameters, print_trainable_parameters

import warnings
warnings.filterwarnings("ignore")


def freeze_non_lora_parameters(model):
    """Freeze all non-LoRA parameters"""
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def train_lora(configs, gpus):
    """Main LoRA fine-tuning function"""
    
    # Load datasets
    train_set = KaldiDiarizationDataset(
        data_dir=configs["data"]["train_data_dir"],
        chunk_size=configs["data"]["chunk_size"],
        context_size=configs["data"]["context_recp"],
        input_transform=configs["data"]["feat_type"],
        frame_size=configs["data"]["feat"]["win_length"],
        frame_shift=configs["data"]["feat"]["hop_length"],
        subsampling=configs["data"]["subsampling"],
        rate=configs["data"]["feat"]["sample_rate"],
        label_delay=configs["data"]["label_delay"],
        n_speakers=configs["data"]["num_speakers"],
        use_last_samples=configs["data"]["use_last_samples"],
        shuffle=configs["data"]["shuffle"]
    )
    
    val_set = KaldiDiarizationDataset(
        data_dir=configs["data"]["val_data_dir"],
        chunk_size=configs["data"]["chunk_size"],
        context_size=configs["data"]["context_recp"],
        input_transform=configs["data"]["feat_type"],
        frame_size=configs["data"]["feat"]["win_length"],
        frame_shift=configs["data"]["feat"]["hop_length"],
        subsampling=configs["data"]["subsampling"],
        rate=configs["data"]["feat"]["sample_rate"],
        label_delay=configs["data"]["label_delay"],
        n_speakers=configs["data"]["num_speakers"],
        use_last_samples=configs["data"]["use_last_samples"],
        shuffle=configs["data"]["shuffle"]
    )
    
    datasets = {
        "train": train_set,
        "val": val_set
    }
    
    collate_func = my_collate
    
    # Define model
    print("Initializing base model...")
    model = OnlineTransformerDADiarization(
        n_speakers=configs["data"]["num_speakers"],
        in_size=(2 * configs["data"]["context_recp"] + 1) * configs["data"]["feat"]["n_mels"],
        **configs["model"]["params"],
    )
    
    # Load pre-trained checkpoint
    if configs["training"]["init_ckpt"]:
        print(f"Loading pre-trained checkpoint from {configs['training']['init_ckpt']}")
        ckpt_package = torch.load(configs["training"]["init_ckpt"], map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if "state_dict" in ckpt_package:
            state_dict = ckpt_package["state_dict"]
        else:
            state_dict = ckpt_package
        
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[len('model.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Pre-trained weights loaded successfully!")
    else:
        raise ValueError("init_ckpt must be provided for LoRA fine-tuning!")
    
    # Apply LoRA if enabled
    if configs.get("lora", {}).get("enabled", False):
        print("\nApplying LoRA to model...")
        lora_config = configs["lora"]
        model = apply_lora_to_model(model, lora_config)
        
        # Freeze non-LoRA parameters
        if lora_config.get("freeze_base_model", True):
            freeze_non_lora_parameters(model)
        
        print("\nTrainable parameters after LoRA:")
        print_trainable_parameters(model)
        print()
    else:
        print("LoRA not enabled, training full model")
    
    # Define optimizer - CRITICAL FIX
    if configs.get("lora", {}).get("enabled", False):
        print("\n" + "="*50)
        print("Collecting LoRA parameters for optimizer...")
        print("="*50)
        
        # Collect LoRA parameters directly
        lora_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora_' in name:
                lora_params.append(param)
                print(f"  ✓ {name}: {param.shape}, {param.numel():,} params")
        
        print(f"\nTotal LoRA parameters: {sum(p.numel() for p in lora_params):,}")
        print("="*50 + "\n")
        
        if len(lora_params) == 0:
            raise ValueError("No LoRA parameters found! Something went wrong with LoRA application.")
        
        opt_config = {
            "params": lora_params,
            "lr": configs["training"]["lr"]
        }
    else:
        print("LoRA not enabled, training full model")
        opt_config = {
            "params": model.parameters(),
            "lr": configs["training"]["lr"]
        }
    
    opt_name = configs["training"]["opt"].lower()
    if opt_name == "adam":
        opt = torch.optim.Adam(**opt_config)
    elif opt_name == "sgd":
        opt = torch.optim.SGD(**opt_config)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(**opt_config)
    elif opt_name == "noam":
        opt = torch.optim.Adam(**opt_config, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise NotImplementedError(f"Optimizer {opt_name} not implemented")
    
    # Verify optimizer has parameters
    print("\nOptimizer verification:")
    print(f"Number of parameter groups: {len(opt.param_groups)}")
    for i, group in enumerate(opt.param_groups):
        print(f"  Group {i}: {len(group['params'])} tensors, {sum(p.numel() for p in group['params']):,} parameters")
    print()
    
    # Scheduler
    if configs["training"].get("scheduler"):
        print("Using noam scheduler")
        scheduler = NoamScheduler(
            opt,
            configs["model"]["params"]["n_units"],
            configs["training"]["warm_steps"],
            scale=configs["training"]["schedule_scale"]
        )
    else:
        scheduler = None
    
    # Define the logger
    logger = TensorBoardLogger(
        os.path.dirname(configs["log"]["log_dir"]),
        configs["log"]["model_name"]
    )
    configs["log"]["log_dir"] = logger.log_dir
    print(f"Experiment directory: {configs['log']['log_dir']}")
    
    os.makedirs(configs["log"]["log_dir"], exist_ok=True)
    
    # Save config
    with open(configs["log"]["log_dir"] + "/config.yaml", "w") as f:
        yaml.dump(configs, f)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val/obj_metric",
            patience=configs["training"]["early_stop_epoch"],
            verbose=True,
            mode="min"
        ),
        ModelCheckpoint(
            logger.log_dir,
            monitor="val/obj_metric",
            save_top_k=configs["log"]["save_top_k"],
            mode="min",
            save_last=True
        )
    ]
    
    # Define the training setup
    spk_dia_main = SpeakerDiarization(
        hparams=configs,
        model=model,
        datasets=datasets,
        opt=opt,
        scheduler=scheduler,
        collate_func=collate_func
    )
    
    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=configs["training"]["max_epochs"],
        callbacks=callbacks,
        accelerator="gpu" if gpus else "cpu",
        devices=gpus if gpus else "auto",
        strategy=configs["training"]["dist_strategy"] if configs["training"]["dist_strategy"] else "auto",
        accumulate_grad_batches=configs["training"]["grad_accm"],
        logger=logger,
        gradient_clip_val=configs["training"]["grad_clip"],
        check_val_every_n_epoch=configs["training"]["val_interval"],
        **configs["debug"]
    )
    
    # Start training
    print("\n" + "="*50)
    print("Starting LoRA Fine-tuning")
    print("="*50 + "\n")
    trainer.fit(spk_dia_main)
    
    best_path = trainer.checkpoint_callback.best_model_path
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    print(f"Best checkpoint: {best_path}")
    print(f"All checkpoints saved in: {os.path.dirname(best_path)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--configs', help='Configuration file path', required=True)
    parser.add_argument('--gpus', default=None, help='Device used for training')
    
    setup = parser.parse_args()
    
    with open(setup.configs, "r") as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)
    
    # Freeze seed for reproducibility
    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    train_lora(configs, gpus=setup.gpus)