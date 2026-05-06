import os
import shutil
import argparse
import glob

def clean_orphaned_data(json_input_dir, json_data_dir):
    # Find all batch directories in json_data (e.g., .../json_data/batch_1)
    data_batches = glob.glob(os.path.join(json_data_dir, "batch_*"))

    if not data_batches:
        print(f"No batch directories found in '{json_data_dir}'")
        return

    total_deleted = 0

    for data_batch_path in data_batches:
        # Ensure it's actually a directory
        if not os.path.isdir(data_batch_path):
            continue
            
        batch_name = os.path.basename(data_batch_path)
        input_batch_path = os.path.join(json_input_dir, batch_name)

        # If the corresponding input batch folder is entirely missing, skip to avoid accidentally deleting everything
        if not os.path.exists(input_batch_path):
            print(f"Warning: Input directory '{input_batch_path}' is missing. Skipping cleanup for {batch_name}.")
            continue

        # 1. Build a set of valid structure names from the input batch
        # For example, "000200.json" becomes "000200"
        valid_structure_names = set()
        for input_file in glob.glob(os.path.join(input_batch_path, "*.json")):
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            valid_structure_names.add(base_name)

        # 2. Iterate through all items in the corresponding data batch
        for item in os.listdir(data_batch_path):
            item_path = os.path.join(data_batch_path, item)
            
            # We are only looking for subdirectories (e.g., '000200')
            if os.path.isdir(item_path):
                # If the subdirectory name doesn't exist in our set of valid inputs, delete it
                if item not in valid_structure_names:
                    print(f"Deleting orphaned directory: {item_path}")
                    try:
                        shutil.rmtree(item_path)
                        total_deleted += 1
                    except Exception as e:
                        print(f"  Error deleting {item_path}: {e}")

    print(f"\nCleanup complete. Successfully deleted {total_deleted} orphaned structure directories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete json_data directories that no longer have a corresponding json_input file.")
    
    # Add CLI flags
    parser.add_argument(
        "-i", "--input_dir", 
        required=True, 
        help="Base directory containing the json_input batches (e.g., /pub/absara/datasets/ASD/af3/json_input)"
    )
    parser.add_argument(
        "-d", "--data_dir", 
        required=True, 
        help="Base directory containing the json_data batches (e.g., /pub/absara/datasets/ASD/af3/json_data)"
    )
    
    args = parser.parse_args()
    
    # Validate directories exist before running
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
    elif not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
    else:
        print("Scanning for orphaned directories...\n")
        clean_orphaned_data(args.input_dir, args.data_dir)