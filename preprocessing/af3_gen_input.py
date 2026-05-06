import pandas as pd
import json
import os
import argparse
import glob  # Added to help search for old files

def chunk_list(lst, n):
    """Divides a list into n fairly equal-sized chunks."""
    if not lst:
        return []
    # If there are fewer items than requested batches, reduce n to the number of items
    n = min(n, len(lst))
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def generate_batched_af3_inputs(csv_file_path, output_dir, num_batches):
    # Read the CSV file. 
    # Enforcing 'library_id' as string preserves any leading zeros.
    df = pd.read_csv(csv_file_path, dtype={'library_id': str})
    
    # Filter for rows where the 'wild_type' column is True.
    wild_type_df = df[df['wild_type'].astype(str).str.lower() == 'true']
    
    if wild_type_df.empty:
        print("No valid 'wild_type' entries found to process.")
        return

    # Ensure the main output directory exists before trying to clean it
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # CLEANUP: Wipe old .json files in the output directory
    # ---------------------------------------------------------
    print(f"Scanning for old .json files to wipe in '{output_dir}'...")
    old_jsons = glob.glob(os.path.join(output_dir, "**", "*.json"), recursive=True)
    
    removed_count = 0
    for old_file in old_jsons:
        try:
            os.remove(old_file)
            removed_count += 1
        except OSError as e:
            print(f"Warning: Could not remove {old_file}: {e}")
            
    if removed_count > 0:
        print(f"Successfully wiped {removed_count} old .json file(s).")
    else:
        print("No old .json files found to wipe.")
    # ---------------------------------------------------------

    # Store tuples of (library_id, payload) so we can batch them
    all_entries = []
    
    # Iterate over the filtered rows
    for index, row in wild_type_df.iterrows():
        library_id = row['library_id']
        sequences = []
        
        # Extract sequences, drop hyphens, and append them as distinct AF3 protein entities
        heavy_seq = str(row['heavy_sequence']).strip().replace("-", "") if pd.notna(row['heavy_sequence']) else ""
        if heavy_seq:
            sequences.append({
                "protein": {
                    "id": "H",
                    "sequence": heavy_seq
                }
            })
            
        light_seq = str(row['light_sequence']).strip().replace("-", "") if pd.notna(row['light_sequence']) else ""
        if light_seq:
            sequences.append({
                "protein": {
                    "id": "L",
                    "sequence": light_seq
                }
            })
            
        antigen_seq = str(row['antigen_sequence']).strip().replace("-", "") if pd.notna(row['antigen_sequence']) else ""
        if antigen_seq:
            sequences.append({
                "protein": {
                    "id": "A",
                    "sequence": antigen_seq
                }
            })
            
        # Construct the final AF3 JSON payload structure
        af3_payload = {
            "name": library_id,
            "modelSeeds": [1],            # AF3 requires at least one integer seed
            "sequences": sequences,
            "dialect": "alphafold3",      # Required by AF3 locally
            "version": 3                  # Based on AF3 latest documentation
        }
        
        all_entries.append((library_id, af3_payload))
        
    # Split the structures into the requested number of batches
    batches = chunk_list(all_entries, num_batches)
    
    total_files_written = 0
    
    # Write the batched files to their respective subdirectories
    for i, batch in enumerate(batches):
        batch_num = i + 1  # 1-indexed batch number for the folder name
        batch_dir = os.path.join(output_dir, f"batch_{batch_num}")
        os.makedirs(batch_dir, exist_ok=True)
        
        for library_id, payload in batch:
            # Define the output filepath (e.g., output_dir/batch_1/000200.json)
            output_filename = os.path.join(batch_dir, f"{library_id}.json")
            
            with open(output_filename, 'w') as json_file:
                json.dump(payload, json_file, indent=4)
                
            total_files_written += 1
            
        print(f"Generated {len(batch)} AF3 JSON files in '{batch_dir}'.")
        
    print(f"\nSuccessfully generated {total_files_written} total AF3 JSON files across {len(batches)} batch subdirectories.")

if __name__ == "__main__":
    # Set up argparse for CLI usage
    parser = argparse.ArgumentParser(description="Convert protein sequence CSV to batched AlphaFold 3 JSON subdirectories.")
    
    # Add CLI flags
    parser.add_argument("-i", "--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Main directory to save the batch subdirectories.")
    parser.add_argument("-b", "--batches", type=int, required=True, help="Number of batch subdirectories to create.")
    
    # Parse the arguments provided by the user
    args = parser.parse_args()
    
    # Validate the number of batches
    if args.batches < 1:
        parser.error("The number of batches (-b) must be at least 1.")
    
    # Run the generator if the input file exists
    if os.path.exists(args.input):
        generate_batched_af3_inputs(args.input, args.output_dir, args.batches)
    else:
        print(f"Error: Input file '{args.input}' not found.")