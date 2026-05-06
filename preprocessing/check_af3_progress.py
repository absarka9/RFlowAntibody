import os
import glob
import argparse

def check_msa_completion(input_dir, output_dir):
    # Find all batch directories in the input directory
    batch_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("batch_")]
    
    # Sort them numerically (e.g., batch_1, batch_2, ... batch_10)
    batch_dirs.sort(key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0)
    
    if not batch_dirs:
        print(f"No 'batch_X' subdirectories found in {input_dir}")
        return

    # PRE-SCAN: Find all completed data files across ANY directory in the output_dir
    completed_ids = set()
    if os.path.exists(output_dir):
        # os.walk will search through the main output dir and all batch subdirectories
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith("_data.json"):
                    # Extract just the base identifier (e.g., '000200_data.json' -> '000200')
                    base_id = file.replace("_data.json", "")
                    completed_ids.add(base_id)

    total_expected = 0
    total_completed = 0
    
    # Print table header
    print(f"\n{'Batch Name':<15} | {'Completed / Expected':<22} | {'Status'}")
    print("-" * 60)
    
    for batch in batch_dirs:
        input_batch_path = os.path.join(input_dir, batch)
        
        # Get all input .json files in this specific input batch
        input_files = glob.glob(os.path.join(input_batch_path, "*.json"))
        expected_count = len(input_files)
        total_expected += expected_count
        
        completed_count = 0
        for input_file in input_files:
            # Extract the base identifier from the input filename (e.g., '000200.json' -> '000200')
            base_id = os.path.basename(input_file).replace(".json", "")
            
            # Check if this ID exists anywhere in our master list of completed files
            if base_id in completed_ids:
                completed_count += 1
        
        total_completed += completed_count
        
        # Determine visual status
        if expected_count == 0:
            status = "Empty"
        elif completed_count == expected_count:
            status = "✅ Complete"
        elif completed_count > 0:
            status = "⏳ In Progress"
        else:
            status = "❌ Not Started"
            
        print(f"{batch:<15} | {completed_count:>9} / {expected_count:<10} | {status}")
        
    # Print table footer with totals
    print("-" * 60)
    overall_percentage = round((total_completed / total_expected) * 100, 2) if total_expected > 0 else 0
    print(f"{'TOTAL':<15} | {total_completed:>9} / {total_expected:<10} | 📊 {overall_percentage}% Overall\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check completion of AF3 MSA generation across batches (cross-directory).")
    
    parser.add_argument("-i", "--input_dir", required=True, help="Base directory containing the input batch subdirectories.")
    parser.add_argument("-o", "--output_dir", required=True, help="Base directory containing the output batch subdirectories.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
    else:
        check_msa_completion(args.input_dir, args.output_dir)