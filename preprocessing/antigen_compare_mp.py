import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
from Bio import Align

def get_subset_a(df):
    """
    Subset A: 
    1. library_id must have at least 8 rows where wild_type_length == True.
    2. The row itself must have wild_type == True.
    """
    length_col = 'wild_type_length' if 'wild_type_length' in df.columns else 'modal_length'
    
    if length_col not in df.columns:
        raise KeyError(f"Neither 'wild_type_length' nor 'modal_length' found in columns: {df.columns}")
    
    lib_length_counts = df[df[length_col] == True].groupby('library_id').size()
    valid_library_ids = lib_length_counts[lib_length_counts >= 8].index
    
    filtered_df = df[(df['library_id'].isin(valid_library_ids)) & (df['wild_type'] == True)]
    return filtered_df

def get_subset_b(df):
    """
    Subset B: 
    All rows where wild_type == True.
    """
    return df[df['wild_type'] == True]

def align_worker(args):
    """
    Worker function to compute alignment for a single pair of sequences.
    """
    i, j, seq1, seq2, self_score1, self_score2 = args
    
    if not isinstance(seq1, str) or not isinstance(seq2, str) or len(seq1) == 0 or len(seq2) == 0:
        return i, j, 0.0, 0.0
        
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    
    alignments = aligner.align(seq1, seq2)
    
    try:
        best_alignment = alignments[0]
        
        # 1. Sequence Similarity
        raw_score = best_alignment.score
        max_possible = max(self_score1, self_score2)
        norm_score = raw_score / max_possible if max_possible > 0 else 0
        
        # 2. Sequence Identity
        aligned_seq1 = best_alignment[0]
        aligned_seq2 = best_alignment[1]
        matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
        identity = matches / max(len(seq1), len(seq2))
        
    except (IndexError, StopIteration):
        norm_score = 0.0
        identity = 0.0
        
    return i, j, norm_score, identity

def calculate_matrices(unique_antigens, cores, has_ref_seq=False):
    """
    Calculates all-by-all similarity and identity matrices using multiprocessing.
    """
    n_antigens = len(unique_antigens)
    
    if n_antigens == 0:
        return None, None
        
    print(f"Pre-computing self-alignment scores for {n_antigens} sequences...")
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    self_scores = [aligner.score(seq, seq) for seq in unique_antigens]
    
    tasks = []
    for i in range(n_antigens):
        for j in range(i, n_antigens):
            tasks.append((i, j, unique_antigens[i], unique_antigens[j], self_scores[i], self_scores[j]))
            
    total_tasks = len(tasks)
    print(f"Distributing {total_tasks} alignment tasks across {cores} cores...")
    
    similarity_matrix = np.zeros((n_antigens, n_antigens))
    identity_matrix = np.zeros((n_antigens, n_antigens))
    
    with mp.Pool(processes=cores) as pool:
        for idx, result in enumerate(pool.imap_unordered(align_worker, tasks), 1):
            i, j, norm_score, identity = result
            
            similarity_matrix[i, j] = norm_score
            similarity_matrix[j, i] = norm_score 
            
            identity_matrix[i, j] = identity
            identity_matrix[j, i] = identity
            
            if idx % 5000 == 0 or idx == total_tasks:
                print(f"  Processed {idx}/{total_tasks} alignments ({(idx/total_tasks)*100:.1f}%)")

    labels = [f"Ag_{i}" for i in range(n_antigens)]
    # If a user reference sequence was appended, rename the last label
    if has_ref_seq:
        labels[-1] = "User_Reference_Seq"
        
    sim_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    id_df = pd.DataFrame(identity_matrix, index=labels, columns=labels)
    
    return sim_df, id_df

def generate_plots(matrix_df, output_base_path, metric_name, subset_name, has_ref_seq=False):
    """
    Generates and saves a standard heatmap, computing both the off-diagonal average
    and average max values, marking both directly on the colorbar legend.
    """
    if matrix_df is None or len(matrix_df) == 0:
        print(f"Error: The {metric_name} matrix for {subset_name} is empty. No plots to generate.")
        return
        
    os.makedirs(os.path.dirname(os.path.abspath(output_base_path)), exist_ok=True)
    metric_lower = metric_name.lower()
    
    vals = matrix_df.values.copy()
    if len(vals) > 1:
        np.fill_diagonal(vals, np.nan)
        avg_val = np.nanmean(vals)
        row_maxes = np.nanmax(vals, axis=1)
        avg_max_val = np.nanmean(row_maxes)
    else:
        avg_val = vals[0, 0] if len(vals) == 1 else 0.0
        avg_max_val = vals[0, 0] if len(vals) == 1 else 0.0
        
    print(f"--> Calculated {subset_name} Average {metric_name}: {avg_val:.4f}")
    print(f"--> Calculated {subset_name} Average Max {metric_name}: {avg_max_val:.4f}")
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        matrix_df, 
        cmap="viridis", 
        vmin=0, 
        vmax=1, 
        square=True,
        xticklabels=False,  
        yticklabels=False   
    )
    
    # If a reference sequence is included, draw a subtle white dotted line to visually separate the last row/col
    if has_ref_seq and len(matrix_df) > 1:
        n_rows = len(matrix_df)
        ax.axhline(n_rows - 1, color='white', linewidth=1.5, linestyle=':')
        ax.axvline(n_rows - 1, color='white', linewidth=1.5, linestyle=':')
    
    if not np.isnan(avg_val) and not np.isnan(avg_max_val):
        cbar = ax.collections[0].colorbar
        
        cbar.ax.axhline(avg_max_val, color='red', linewidth=2, linestyle='--')
        cbar.ax.text(1.15, avg_max_val, f'Avg Max\n{avg_max_val:.3f}', 
                     va='center', ha='left', color='red', fontweight='bold', 
                     transform=cbar.ax.get_yaxis_transform())
                     
        cbar.ax.axhline(avg_val, color='dodgerblue', linewidth=2, linestyle='--')
        cbar.ax.text(1.15, avg_val, f'Avg\n{avg_val:.3f}', 
                     va='center', ha='left', color='dodgerblue', fontweight='bold', 
                     transform=cbar.ax.get_yaxis_transform())
    
    plt.title(f"Antigen Sequence {metric_name} Heatmap ({subset_name.replace('_', ' ').title()})")
    
    heatmap_path = f"{output_base_path}_{subset_name}_{metric_lower}_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to: {heatmap_path}")
    plt.close()

def process_subset(unique_antigens, subset_name, args, has_ref_seq=False):
    """
    Helper function to calculate matrices and generate plots for a given subset.
    """
    print(f"\n======================================")
    print(f"Processing {subset_name} ({len(unique_antigens)} unique antigens)")
    print(f"======================================")
    
    if len(unique_antigens) == 0:
        print(f"No sequences found for {subset_name}. Skipping.")
        return

    sim_matrix, id_matrix = calculate_matrices(unique_antigens, cores=args.cores, has_ref_seq=has_ref_seq)
    
    sim_csv_path = f"{args.output}_{subset_name}_similarity_matrix.csv"
    id_csv_path = f"{args.output}_{subset_name}_identity_matrix.csv"
    sim_matrix.to_csv(sim_csv_path)
    id_matrix.to_csv(id_csv_path)
    print(f"Saved matrices to:\n  {sim_csv_path}\n  {id_csv_path}")
    
    print("Generating plots...")
    generate_plots(sim_matrix, args.output, metric_name="Similarity", subset_name=subset_name, has_ref_seq=has_ref_seq)
    generate_plots(id_matrix, args.output, metric_name="Identity", subset_name=subset_name, has_ref_seq=has_ref_seq)

def main():
    parser = argparse.ArgumentParser(description="Multicore antigen similarity/identity analysis.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, help="Base path/prefix for outputs.")
    parser.add_argument("--ref_seq", type=str, default="", help="Arbitrary reference sequence to append as the last row/col.")
    
    max_cores = mp.cpu_count()
    default_cores = max(1, max_cores - 1)
    parser.add_argument("--cores", type=int, default=default_cores, 
                        help=f"Number of CPU cores to use (default: {default_cores} on this machine)")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    
    # Extract unique antigens
    unique_antigens_a = list(get_subset_a(df)['antigen_sequence'].dropna().unique())
    unique_antigens_b = list(get_subset_b(df)['antigen_sequence'].dropna().unique())
    
    has_ref_seq = False
    if args.ref_seq:
        has_ref_seq = True
        print(f"Appending user-provided reference sequence ({len(args.ref_seq)} AA).")
        # Append to the end of the arrays
        if len(unique_antigens_a) > 0:
            unique_antigens_a.append(args.ref_seq)
        if len(unique_antigens_b) > 0:
            unique_antigens_b.append(args.ref_seq)
            
    # Process both subsets
    process_subset(unique_antigens_a, "subset_A", args, has_ref_seq)
    process_subset(unique_antigens_b, "subset_B", args, has_ref_seq)
    
    print("\nAll done!")

if __name__ == "__main__":
    main()