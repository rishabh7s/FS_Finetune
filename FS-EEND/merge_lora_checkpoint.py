import torch
import os


def pure_merge(base_ckpt_path, lora_ckpt_path, output_path, scaling=2.0):
    if not os.path.exists(lora_ckpt_path):
        print(f"Error: LoRA checkpoint not found at {lora_ckpt_path}")
        return

    print("--- Starting Pure Logic Merge ---")
    
    # Load state dicts
    lora_data = torch.load(lora_ckpt_path, map_location='cpu', weights_only=False)
    lora_state = lora_data.get('state_dict', lora_data)
    
    merged_state = {}
    
    # We iterate through the LoRA state because it contains the structural maps
    for key in list(lora_state.keys()):
        clean_key = key.replace('model.', '')

        # --- LOGIC 1: TRANSFORMER LINEAR LAYERS ---
        if '.linear.weight' in key:
            target_key = clean_key.replace('.linear.weight', '.weight')
            lora_a_key = key.replace('.linear.weight', '.lora.lora_A')
            lora_b_key = key.replace('.linear.weight', '.lora.lora_B')
            
            W_base = lora_state[key]
            A = lora_state[lora_a_key]
            B = lora_state[lora_b_key]
            
            merged_state[target_key] = W_base + (B @ A) * scaling
            print(f"Collapsed: {target_key}")

        # --- LOGIC 2: SELF-ATTENTION ---
        elif '.self_attn.attn.' in key and '.weight' in key:
            target_key = clean_key.replace('.attn.', '.')
            
            if 'in_proj' in key:
                lora_a_key = key.replace('.attn.in_proj_weight', '.lora_in_proj.lora_A')
                lora_b_key = key.replace('.attn.in_proj_weight', '.lora_in_proj.lora_B')
            else:
                lora_a_key = key.replace('.attn.out_proj.weight', '.lora_out_proj.lora_A')
                lora_b_key = key.replace('.attn.out_proj.weight', '.lora_out_proj.lora_B')
            
            W_base = lora_state[key]
            A = lora_state[lora_a_key]
            B = lora_state[lora_b_key]
            
            merged_state[target_key] = W_base + (B @ A) * scaling
            print(f"Collapsed: {target_key}")

        # --- LOGIC 3: CNN ---
        elif 'cnn.conv.weight' in key:
                    target_key = 'cnn.weight'
                    W_base = lora_state[key]         # Shape: [256, 256, 19]
                    A = lora_state['model.cnn.lora_A'] # Shape: [8, 4864]
                    B = lora_state['model.cnn.lora_B'] # Shape: [256, 8]
                    
                    # 1. Matrix multiply B and A -> Result shape: [256, 4864]
                    lora_update = B @ A
                    
                    # 2. Reshape the update to match W_base -> [256, 256, 19]
                    # .view() or .reshape() will organize the 4864 back into (256, 19)
                    lora_update = lora_update.view(W_base.shape)
                    
                    # 3. Add to base
                    merged_state[target_key] = W_base + (lora_update * scaling)
                    print(f"Collapsed CNN:    {target_key} (Reshaped 4864 -> {W_base.shape[1:]})")
        # --- LOGIC 4: BIASES ---
        elif '.linear.bias' in key:
            merged_state[clean_key.replace('.linear.bias', '.bias')] = lora_state[key]
        elif '.attn.in_proj_bias' in key:
            merged_state[clean_key.replace('.attn.in_proj_bias', '.in_proj_bias')] = lora_state[key]
        elif '.attn.out_proj.bias' in key:
            merged_state[clean_key.replace('.attn.out_proj.bias', '.out_proj.bias')] = lora_state[key]
        elif 'cnn.conv.bias' in key:
            merged_state['cnn.bias'] = lora_state[key]

        # --- LOGIC 5: STATIC WEIGHTS (Norms, BN, PE) ---
        # Exclude all A, B, and the specific wrapped keys handled above
        elif not any(x in key for x in ['.lora.', 'lora_', '.linear.', '.attn.']):
            merged_state[clean_key] = lora_state[key]

    print(f"\nSaving {len(merged_state)} keys to {output_path}...")
    torch.save(merged_state, output_path)
    print("Merge Complete!")

if __name__ == "__main__":
    # Update these paths to your actual file names
    pure_merge(
        base_ckpt_path=r"C:\Users\Rishabh Singh\Downloads\FineTuning_LS-EEND\FS-EEND\FS-EEND\ckpt\simu\FS-EEND_ch_91_100epo_avg_model.ckpt", 
        lora_ckpt_path=r"C:\Users\Rishabh Singh\Downloads\FineTuning_LS-EEND\FS-EEND\FS-EEND\last.ckpt",
        output_path="merged_inference.ckpt"
    )