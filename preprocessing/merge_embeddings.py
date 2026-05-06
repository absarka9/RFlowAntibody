import os
import torch
from pathlib import Path
import argparse

def merge_embedding_dicts(directory_path: str, output_filename: str = "merged_embeddings.pt"):
    """
    Recursively finds all .pt files containing dictionaries in a directory, 
    merges them into a single dictionary, and saves it.
    """
    target_dir = Path(directory_path)
    
    if not target_dir.is_dir():
        raise ValueError(f"Error: The path '{directory_path}' is not a valid directory.")

    output_file_path = target_dir / output_filename
    merged_dict = {}
    processed_files = 0
    duplicate_keys = 0

    print(f"Searching for .pt files in: {target_dir.resolve()}...")

    # Recursively iterate through all .pt files
    for file_path in target_dir.rglob('*.pt'):
        if file_path.resolve() == output_file_path.resolve():
            continue

        try:
            # Set weights_only=False because we are loading dicts, not raw tensors
            file_data = torch.load(file_path, weights_only=False, map_location='cpu')
            
            if not isinstance(file_data, dict):
                print(f"[-] Skipping {file_path.name}: Contains {type(file_data)} instead of a dict.")
                continue
            
            # Merge the current file's dictionary into the master dictionary
            for key, value in file_data.items():
                if key in merged_dict:
                    duplicate_keys += 1
                merged_dict[key] = value
                
            processed_files += 1
            print(f"[+] Loaded: {file_path.name} (Added {len(file_data)} items)")
            
        except Exception as e:
            print(f"[-] Error loading {file_path.name}: {e}")

    if not merged_dict:
        print("\nNo valid dictionary data found to merge.")
        return

    print("\nSaving merged dictionary...")
    
    try:
        # Save the combined dictionary to disk
        torch.save(merged_dict, output_file_path)
        
        print("-" * 30)
        print("Success!")
        print(f"Files processed:     {processed_files}")
        print(f"Total items merged:  {len(merged_dict)}")
        if duplicate_keys > 0:
            print(f"Warning: Overwrote {duplicate_keys} duplicate IDs/keys.")
        print(f"Saved output to:     {output_file_path.resolve()}")
        
    except Exception as e:
        print(f"\n[-] Error during saving: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PyTorch .pt dictionary files.")
    parser.add_argument("directory", type=str, help="Target directory to search.")
    parser.add_argument("--output", type=str, default="merged_embeddings.pt", help="Output filename.")
    
    args = parser.parse_args()
    merge_embedding_dicts(args.directory, args.output)