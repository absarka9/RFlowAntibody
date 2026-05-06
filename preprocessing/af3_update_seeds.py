import json
import glob
import os
import argparse

def update_seeds(base_dir, num_seeds):
    # Use '**' and recursive=True to search all subdirectories within each batch folder
    search_pattern = os.path.join(base_dir, "batch_*", "**", "*_data.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"No '_data.json' files found matching {search_pattern}")
        return

    # Generate the list of seeds: e.g., if num_seeds is 5, this creates [1, 2, 3, 4, 5]
    seed_list = list(range(1, num_seeds + 1))
    
    updated_count = 0

    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Update the seeds array
            data['modelSeeds'] = seed_list
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
                
            updated_count += 1
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"Successfully updated {updated_count} files to use {num_seeds} seeds: {seed_list}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the number of modelSeeds in AlphaFold 3 _data.json files.")
    
    # Add CLI flags
    parser.add_argument(
        "-d", "--dir", 
        required=True, 
        help="Base directory containing the batch subdirectories (e.g., /pub/absara/datasets/ASD/af3/json_data)"
    )
    parser.add_argument(
        "-s", "--seeds", 
        type=int, 
        required=True, 
        help="Number of random seeds to set for each structure."
    )
    
    args = parser.parse_args()
    
    # Validate the number of seeds
    if args.seeds < 1:
        parser.error("The number of seeds (-s) must be at least 1.")
        
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' not found.")
    else:
        print(f"Scanning {args.dir} for data payloads...")
        update_seeds(args.dir, args.seeds)