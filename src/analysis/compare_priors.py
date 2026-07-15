import argparse
import pandas as pd
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser
from transformers import EsmModel, EsmTokenizer

# Attempt to import MINT utilities if they exist in the environment
try:
    from mint.helpers.extract import load_config, MINTWrapper, CollateFn
except ImportError:
    MINTWrapper = None
    CollateFn = None

def load_wt_sequence_from_csv(csv_path, library_id):
    """Finds the WT sequence for a specific library_id in the CSV."""
    # Prevent DtypeWarning and safely handle mixed types
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Cast CSV column to string, strip decimals if read as float, and pad to 6 digits
    df['library_id'] = df['library_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
    lib_id_str = str(library_id).zfill(6) 
    
    # Safely handle the wild_type boolean check
    df['wild_type_str'] = df['wild_type'].astype(str).str.lower().str.strip()
    wt_rows = df[(df['library_id'] == lib_id_str) & (df['wild_type_str'].isin(['true', '1', '1.0']))]
    
    if len(wt_rows) == 0:
        raise ValueError(f"No row found with library_id='{lib_id_str}' and wild_type==True in {csv_path}")
    if len(wt_rows) > 1:
        print(f"Warning: Found {len(wt_rows)} wild-types for library {lib_id_str}. Using the first one.")
        
    row = wt_rows.iloc[0]
    
    chains = []
    for col in ['heavy_sequence', 'light_sequence', 'antigen_sequence']:
        if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != "":
            chains.append(str(row[col]).strip())
            
    flat_seq = "".join(chains)
    print(f"Loaded WT Complex for Library {lib_id_str} (Total Length: {len(flat_seq)})")
    return flat_seq, chains

def get_cif_path_from_yaml(yaml_path, library_id):
    """Extracts the AF3 .cif path from the yaml directory map."""
    with open(yaml_path, 'r') as f:
        dir_map = yaml.safe_load(f)
        
    dir_map_str_keys = {str(k).zfill(6): v for k, v in dir_map.items()}
    lib_id_str = str(library_id).zfill(6) 
    
    if lib_id_str not in dir_map_str_keys:
        raise KeyError(f"Library ID {lib_id_str} not found in YAML map {yaml_path}")
        
    cif_path = Path(dir_map_str_keys[lib_id_str])
    if not cif_path.exists():
        raise FileNotFoundError(f"Mapped CIF file does not exist: {cif_path}")
        
    return str(cif_path)

def apply_apc(attn_matrix):
    """Applies Average Product Correction (APC) to reveal sparse inter-chain contacts."""
    attn_matrix = 0.5 * (attn_matrix + attn_matrix.T)
    row_sum = np.sum(attn_matrix, axis=1, keepdims=True)
    col_sum = np.sum(attn_matrix, axis=0, keepdims=True)
    total_sum = np.sum(attn_matrix)
    
    apc = (row_sum * col_sum) / total_sum
    corrected_attn = attn_matrix - apc
    
    np.fill_diagonal(corrected_attn, 0)
    return np.clip(corrected_attn, 0, None)

def get_esm_attention_map(flat_seq, device='cuda'):
    """Extracts L x L attention maps from base ESM-2, mirroring MINT output structure."""
    print("Computing ESM-2 Attention Maps...")
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, output_attentions=True).to(device)
    model.eval()

    inputs = tokenizer(flat_seq, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # (Layers, Heads, L+2, L+2)
    attns = torch.stack(outputs.attentions).squeeze(1) 
    L_padded = attns.shape[-1]
    attns_flat = attns.reshape(-1, L_padded, L_padded)
    
    # Raw Max
    raw_max_tensor, _ = attns_flat.max(dim=0)
    
    # Thresholded Z-Max
    mean_heads = attns_flat.mean(dim=0, keepdim=True)
    std_heads = attns_flat.std(dim=0, keepdim=True)
    z_attns = (attns_flat - mean_heads) / (std_heads + 1e-5)
    z_max_tensor, _ = z_attns.max(dim=0)
    thresholded_z_tensor = torch.clamp(z_max_tensor - 7.0, min=0)
    
    # Strip <cls> and <eos>
    raw_max = raw_max_tensor[1:-1, 1:-1].cpu().numpy()
    thresholded_z_max = thresholded_z_tensor[1:-1, 1:-1].cpu().numpy()
    
    apc_max = apply_apc(raw_max)
    
    return raw_max, apc_max, thresholded_z_max

def get_mint_attention_map(chains, config_path=None, ckpt_path=None, device='cuda'):
    """Extracts Raw Max, APC, and Thresholded Z-Score attention maps from MINT."""
    if MINTWrapper is None or CollateFn is None:
        raise ImportError("MINT dependencies could not be imported. Check your environment.")
        
    print("Computing MINT Attention Maps...")
    cfg = load_config(config_path) if config_path else None
    wrapper = MINTWrapper(cfg, ckpt_path, sep_chains=True, device=device)
    wrapper.eval()
    
    collater = CollateFn()
    batch = [tuple(chains)] 
    tokenized_chains, tokenized_c_ids = collater(batch)
    
    tokenized_chains = tokenized_chains.to(device)
    tokenized_c_ids = tokenized_c_ids.to(device)
    
    with torch.no_grad():
        out = wrapper.model(
            tokenized_chains, 
            tokenized_c_ids, 
            repr_layers=[cfg.encoder_layers], 
            need_head_weights=True
        )
        
    attns = out["attentions"][0] 
    L = attns.shape[-1]
    attns_flat = attns.reshape(-1, L, L)
    
    # 1. Raw Max
    raw_max_tensor, _ = attns_flat.max(dim=0)
    
    # 2. Thresholded Z-Max
    mean_heads = attns_flat.mean(dim=0, keepdim=True)
    std_heads = attns_flat.std(dim=0, keepdim=True)
    z_attns = (attns_flat - mean_heads) / (std_heads + 1e-5)
    z_max_tensor, _ = z_attns.max(dim=0)
    thresholded_z_tensor = torch.clamp(z_max_tensor - 7.0, min=0)
    
    mask = (
        (~tokenized_chains.eq(wrapper.model.cls_idx)) &
        (~tokenized_chains.eq(wrapper.model.eos_idx)) &
        (~tokenized_chains.eq(wrapper.model.padding_idx))
    )[0] 
    
    # Mask, subset, and move to CPU
    raw_max = raw_max_tensor[mask][:, mask].cpu().numpy()
    thresholded_z_max = thresholded_z_tensor[mask][:, mask].cpu().numpy()
    
    # 3. APC Max (applied to raw max)
    apc_max = apply_apc(raw_max)
    
    return raw_max, apc_max, thresholded_z_max

def get_af3_distogram(cif_path):
    """Parses a .cif file and calculates C-alpha spatial proximity."""
    print(f"Loading AF3 structure from: {cif_path}")
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('AF3_model', cif_path)
    
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ': 
                    if 'CA' in residue:
                        ca_coords.append(residue['CA'].get_coord())
                        
    ca_coords = torch.tensor(np.array(ca_coords))
    dist_matrix = torch.cdist(ca_coords, ca_coords)
    proximity_matrix = torch.exp(-dist_matrix / 10.0) 
    
    return proximity_matrix.numpy()

def min_max_normalize(matrix):
    """Normalizes a matrix to the range [0, 1]."""
    denom = np.max(matrix) - np.min(matrix)
    if denom == 0:
        return matrix
    return (matrix - np.min(matrix)) / denom

def plot_attention_comparisons(raw_attn, apc_attn, thresholded_z_attn, distogram, library_id, output_path):
    print("Generating figures...")
    
    # Normalize physical and raw probability matrices
    norm_dist = min_max_normalize(distogram)
    norm_raw = min_max_normalize(raw_attn)
    norm_apc = min_max_normalize(apc_attn)
    # CRITICAL: Do NOT min-max normalize the Z-score matrix, or it will compress cross-attention
    
    fig, axes = plt.subplots(1, 4, figsize=(32, 7))
    fig.suptitle(f"Structural Prior Comparison: Library {library_id}", fontsize=20, y=1.05)
    
    # 1. AF3 Proximity
    sns.heatmap(norm_dist, ax=axes[0], cmap="magma", square=True, cbar_kws={"shrink": 0.75})
    axes[0].set_title("AF3 C-Alpha Spatial Proximity\n(Physical Distance)", fontsize=14)
    axes[0].set_xlabel("Residue Index")
    axes[0].set_ylabel("Residue Index")
    
    # 2. Raw Max
    sns.heatmap(norm_raw, ax=axes[1], cmap="magma", square=True, cbar_kws={"shrink": 0.75})
    axes[1].set_title("Raw Max Attention\n(All Layers & Heads)", fontsize=14)
    axes[1].set_xlabel("Residue Index")
    
    # 3. APC Max
    sns.heatmap(norm_apc, ax=axes[2], cmap="magma", square=True, cbar_kws={"shrink": 0.75})
    axes[2].set_title("APC Corrected Max\n(Background Bias Removed)", fontsize=14)
    axes[2].set_xlabel("Residue Index")
    
    # 4. Thresholded Z-Max
    # Cap vmax at ~15 so extremely strong backbone interactions don't wash out the color scale
    sns.heatmap(thresholded_z_attn, ax=axes[3], cmap="magma", square=True, vmax=15.0, cbar_kws={"shrink": 0.75})
    axes[3].set_title("Thresholded Z-Score Max\n(Z > 7.0)", fontsize=14)
    axes[3].set_xlabel("Residue Index")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to your library CSV")
    parser.add_argument("--yaml_map", type=str, required=True, help="Path to the YAML directory map for AF3 structures")
    parser.add_argument("--library_id", type=str, required=True, help="The library ID to visualize")
    parser.add_argument("--model", type=str, choices=['esm', 'mint'], default='esm', help="Language model to use")
    parser.add_argument("--mint_config", type=str, default=None, help="Path to MINT config JSON")
    parser.add_argument("--mint_ckpt", type=str, default=None, help="Path to MINT checkpoint")
    parser.add_argument("--output", type=str, default="prior_comparison.png")
    args = parser.parse_args()

    flat_seq, chains = load_wt_sequence_from_csv(args.csv, args.library_id)
    cif_path = get_cif_path_from_yaml(args.yaml_map, args.library_id)
    dist_map = get_af3_distogram(cif_path)
    
    if args.model == 'esm':
        raw_attn, apc_attn, z_attn = get_esm_attention_map(flat_seq)
    else:
        raw_attn, apc_attn, z_attn = get_mint_attention_map(chains, args.mint_config, args.mint_ckpt)
        
    if raw_attn.shape != dist_map.shape:
        raise ValueError(f"Shape mismatch! Attention Map is {raw_attn.shape}, but AF3 Distogram is {dist_map.shape}.")
    
    plot_attention_comparisons(raw_attn, apc_attn, z_attn, dist_map, args.library_id, args.output)