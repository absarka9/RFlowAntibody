import pandas as pd
import json
import os
import argparse

def chunk_list(lst, n):
    """Divides a list into n fairly equal-sized chunks."""
    if not lst:
        return []
    # If there are fewer items than requested batches, reduce n to the number of items
    n = min(n, len(lst))
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def generate_batched_protenix_inputs(csv_file_path, output_prefix, num_batches):
    # Read the CSV file. 
    # Enforcing 'library_id' as string preserves any leading zeros.
    df = pd.read_csv(csv_file_path, dtype={'library_id': str})
    
    # Filter for rows where the 'wild_type' column is True.
    wild_type_df = df[df['wild_type'].astype(str).str.lower() == 'true']
    
    # This list will hold all of our sequence objects
    all_protenix_entries = []
    
    # Iterate over the filtered rows
    for index, row in wild_type_df.iterrows():
        library_id = row['library_id']
        sequences = []
        
        # Extract sequences and append them as distinct protein chains
        if pd.notna(row['heavy_sequence']) and str(row['heavy_sequence']).strip():
            sequences.append({
                "proteinChain": {
                    "sequence": str(row['heavy_sequence']).strip(),
                    "count": 1,
                    "id": ["H"]
                }
            })
            
        if pd.notna(row['light_sequence']) and str(row['light_sequence']).strip():
            sequences.append({
                "proteinChain": {
                    "sequence": str(row['light_sequence']).strip(),
                    "count": 1,
                    "id": ["L"]
                }
            })
            
        if pd.notna(row['antigen_sequence']) and str(row['antigen_sequence']).strip():
            sequences.append({
                "proteinChain": {
                    "sequence": str(row['antigen_sequence']).strip(),
                    "count": 1,
                    "id": ["A"]
                }
            })
            
        # Structure the payload as an object and add it to our main list
        entry_object = {
            "sequences": sequences,
            "name": library_id
        }
        
        all_protenix_entries.append(entry_object)
        
    if not all_protenix_entries:
        print("No valid entries found to process.")
        return

    # Split the main list into the requested number of batches
    batches = chunk_list(all_protenix_entries, num_batches)
    
    # Save each batch to its own JSON file
    for i, batch in enumerate(batches):
        batch_num = i + 1  # 1-indexed batch number for the filename
        output_filename = f"{output_prefix}_{batch_num}.json"
        
        with open(output_filename, 'w') as json_file:
            json.dump(batch, json_file, indent=4)
            
        print(f"Successfully generated {output_filename} containing {len(batch)} entries.")

if __name__ == "__main__":
    # Set up argparse for CLI usage
    parser = argparse.ArgumentParser(description="Convert protein sequence CSV to batched Protenix JSON format.")
    
    # Add CLI flags
    parser.add_argument("-i", "--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output_prefix", required=True, help="Prefix for the output JSON files (e.g., 'batch_output').")
    parser.add_argument("-b", "--batches", type=int, required=True, help="Number of batched JSON files to produce.")
    
    # Parse the arguments provided by the user
    args = parser.parse_args()
    
    # Validate the number of batches
    if args.batches < 1:
        parser.error("The number of batches (-b) must be at least 1.")
    
    # Run the generator if the input file exists
    if os.path.exists(args.input):
        generate_batched_protenix_inputs(args.input, args.output_prefix, args.batches)
    else:
        print(f"Error: Input file '{args.input}' not found.")