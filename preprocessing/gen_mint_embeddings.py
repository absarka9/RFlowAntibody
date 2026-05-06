import torch
import random
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Import your existing MINT classes
from mint.helpers.extract import (
    load_config,
    CollateFn,
    MINTWrapper,
)

class HydratedCSVDataset(Dataset):
    def __init__(self, child_csv_path, master_seq_dict):
        super().__init__()
        self.df = pd.read_csv(child_csv_path)
        
        if "universal_id" not in self.df.columns:
            raise ValueError(f"'universal_id' column missing in {child_csv_path}")
            
        self.uids = self.df["universal_id"].tolist()
        self.master_dict = master_seq_dict

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        uid = self.uids[index]
        chains = self.master_dict[uid]
        return uid, chains


class HydratedCollateFn:
    def __init__(self, truncation_seq_length=None):
        self.base_collate = CollateFn(truncation_seq_length)
        self.alphabet = self.base_collate.alphabet
        self.trunc = truncation_seq_length

    def __call__(self, batches):
        uids = [b[0] for b in batches]
        chains_batch = []
        chain_ids_batch = []
        max_batch_len = 0
        
        for b in batches:
            chains_str_list = b[1] 
            
            variant_tokens = []
            variant_ids = []
            
            for chain_idx, seq_str in enumerate(chains_str_list):
                encoded = self.alphabet.encode("<cls>" + seq_str.replace("J", "L") + "<eos>")
                
                if self.trunc and len(encoded) > self.trunc:
                    start = random.randint(0, len(encoded) - self.trunc)
                    encoded = encoded[start : start + self.trunc]
                
                seq_tensor = torch.tensor(encoded, dtype=torch.int64)
                id_tensor = torch.full((len(encoded),), chain_idx, dtype=torch.int32)
                
                variant_tokens.append(seq_tensor)
                variant_ids.append(id_tensor)
            
            concat_tokens = torch.cat(variant_tokens, dim=-1)
            concat_ids = torch.cat(variant_ids, dim=-1)
            
            chains_batch.append(concat_tokens)
            chain_ids_batch.append(concat_ids)
            
            max_batch_len = max(max_batch_len, concat_tokens.size(0))
            
        batch_size = len(batches)
        padded_chains = torch.full((batch_size, max_batch_len), self.alphabet.padding_idx, dtype=torch.int64)
        padded_ids = torch.full((batch_size, max_batch_len), 0, dtype=torch.int32) 
        
        for i in range(batch_size):
            L = chains_batch[i].size(0)
            padded_chains[i, :L] = chains_batch[i]
            padded_ids[i, :L] = chain_ids_batch[i]
            
        return uids, padded_chains, padded_ids


def load_master_dictionary(master_csv_path):
    print(f"Loading Master CSV into RAM from {master_csv_path}...")
    df_master = pd.read_csv(master_csv_path)
    
    for col in ['universal_id', 'heavy_sequence', 'antigen_sequence']:
        if col not in df_master.columns:
            raise ValueError(f"Missing required column '{col}' in master CSV.")
            
    if 'light_sequence' not in df_master.columns:
        df_master['light_sequence'] = ""
        
    df_master['light_sequence'] = df_master['light_sequence'].fillna("")
    master_dict = {}
    
    for uid, h_seq, l_seq, a_seq in zip(
        df_master['universal_id'], 
        df_master['heavy_sequence'], 
        df_master['light_sequence'], 
        df_master['antigen_sequence']
    ):
        if str(l_seq).strip() == "":
            master_dict[uid] = [str(h_seq), str(a_seq)]
        else:
            master_dict[uid] = [str(h_seq), str(l_seq), str(a_seq)]
            
    print(f"Successfully loaded {len(master_dict)} unique sequences into RAM.")
    return master_dict


def main():
    parser = argparse.ArgumentParser(description="Single-GPU MINT Embedding Harvester for Slurm Arrays")
    parser.add_argument("-m", "--master_csv", required=True)
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-c", "--ckpt", default="/pub/absara/models/mint/mint.ckpt")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}. Exiting.")
        return

    print(f"Found {len(csv_files)} CSV files in {input_dir} to process.")

    # 1. Load Master Dict
    master_dict = load_master_dictionary(args.master_csv)

    # 2. Setup Device and Model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cfg_p = "/pub/absara/models/mint/data/esm2_t33_650M_UR50D.json"
    cfg = load_config(cfg_p)
    
    print("Loading MINT checkpoint...")
    wrapper = MINTWrapper(cfg, args.ckpt, sep_chains=True, device=device)
    wrapper.eval()

    # 3. Process Chunked Files
    with torch.no_grad():
        for csv_path in csv_files:
            embed_out_path = out_dir / f"{csv_path.stem}.pt"
            
            # FAULT TOLERANCE
            if embed_out_path.exists():
                print(f"Skipping {csv_path.name} (output already exists).")
                continue
                
            print(f"\nProcessing {csv_path.name}...")
            
            dataset = HydratedCSVDataset(csv_path, master_dict)
            loader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                collate_fn=HydratedCollateFn(512), 
                shuffle=False
            )

            chunk_embeddings = {}

            for b_idx, (uids, chains, c_ids) in enumerate(loader):
                chains, c_ids = chains.to(device), c_ids.to(device)
                embs = wrapper(chains, c_ids).cpu()
                
                for i, uid in enumerate(uids):
                    chunk_embeddings[uid] = embs[i]

                if b_idx % 10 == 0 and b_idx > 0:
                    print(f"  Processed batch {b_idx}/{len(loader)}")

            torch.save(chunk_embeddings, embed_out_path)
            print(f"✅ Saved {len(chunk_embeddings)} embeddings to {embed_out_path}")

    print("\n✅ Processing complete for this array task.")

if __name__ == "__main__":
    main()