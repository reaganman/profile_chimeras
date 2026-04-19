import argparse
import numpy as np
from Bio import SeqIO
from itertools import product
import csv
import os
import matplotlib.pyplot as plt

def get_k_order_transition_matrix(sequence, k):
    nucs = ['A', 'C', 'G', 'T']
    nuc_to_idx = {n: i for i, n in enumerate(nucs)}
    kmers = [''.join(p) for p in product(nucs, repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(kmers)}
    
    matrix = np.ones((len(kmers), 4)) # Laplace smoothing
    seq = str(sequence).upper()
    
    for i in range(len(seq) - k):
        state = seq[i : i+k]
        next_nuc = seq[i+k]
        if state in kmer_to_idx and next_nuc in nuc_to_idx:
            matrix[kmer_to_idx[state]][nuc_to_idx[next_nuc]] += 1
            
    return matrix / matrix.sum(axis=1)[:, np.newaxis], kmer_to_idx

def calculate_k_log_likelihood(subsequence, matrix, kmer_to_idx, k):
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = str(subsequence).upper()
    log_p = 0.0
    
    for i in range(len(seq) - k):
        state = seq[i : i+k]
        next_nuc = seq[i+k]
        if state in kmer_to_idx and next_nuc in nuc_to_idx:
            prob = matrix[kmer_to_idx[state]][nuc_to_idx[next_nuc]]
            log_p += np.log(prob)
    return log_p

def load_junction_data(tsv_filepath):
    junctions = {}
    with open(tsv_filepath, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            chimera_id = row['Chimera_ID']
            if chimera_id not in junctions:
                junctions[chimera_id] = []
            
            # Grab the source and stop coordinates for plot shading
            junctions[chimera_id].append({
                'fragment_num': int(row['Fragment_Num']),
                'source': row['Source_Genome'],
                'start': int(row['Chimera_Start']),
                'stop': int(row['Chimera_Stop']),
            })
    return junctions

def get_true_parent(chimera_id, window_midpoint, ground_truth):
    """Looks up the true parent for a specific base-pair coordinate."""
    if chimera_id not in ground_truth:
        return None
    for frag in ground_truth[chimera_id]:
        if frag['start'] <= window_midpoint <= frag['stop']:
            return frag['source']
    return None

def get_global_accuracy(chimeras_file, ground_truth, p_models, k, window, step):
    """
    Slides a window across all chimeras and calculates the overall percentage 
    of correct parent predictions.
    """
    from Bio import SeqIO
    total_windows = 0
    correct_predictions = 0
    parent_names = list(p_models.keys())

    for record in SeqIO.parse(chimeras_file, "fasta"):
        seq = record.seq
        chimera_id = record.id
        
        if chimera_id not in ground_truth:
            continue
            
        for start in range(0, len(seq) - window, step):
            midpoint = start + (window // 2)
            subseq = seq[start:start+window]
            
            # Look up the actual parent
            true_parent = get_true_parent(chimera_id, midpoint, ground_truth)
            if not true_parent:
                continue 
                
            # Guess the parent with Markov
            scores = {}
            for parent in parent_names:
                scores[parent] = calculate_k_log_likelihood(subseq, p_models[parent]['matrix'], p_models[parent]['map'], k)
                
            predicted_parent = max(scores, key=scores.get)
            
            # Tally the score
            total_windows += 1
            if predicted_parent == true_parent:
                correct_predictions += 1

    accuracy = (correct_predictions / total_windows) * 100 if total_windows > 0 else 0
    return accuracy, total_windows

def main():
    parser = argparse.ArgumentParser(description="Visualize Relative Parental Markov profiles with Shaded Ground Truth.")
    parser.add_argument("-p", "--parents", required=True)
    parser.add_argument("-c", "--chimeras", required=True)
    parser.add_argument("-j", "--junctions", required=True)
    parser.add_argument("-k", "--order", type=int, default=2, help="Order of the Markov chain (k). Default: 2")
    parser.add_argument("-w", "--window", type=int, default=1000, help="Sliding window size in bp.")
    parser.add_argument("-s", "--step", type=int, default=100, help="Window step size.")
    parser.add_argument("-o", "--outdir", default="plots", help="Directory to save the plots (default: 'plots')")

    args = parser.parse_args()

    # Set up output directory
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(args.outdir)}/")

    # Build Models
    p_models = {}
    parent_names = []
    print(f"Building {args.order}-Order Models...")
    for record in SeqIO.parse(args.parents, "fasta"):
        matrix, mapping = get_k_order_transition_matrix(record.seq, args.order)
        p_models[record.id] = {'matrix': matrix, 'map': mapping}
        parent_names.append(record.id)

    ground_truth = load_junction_data(args.junctions)

    # Define color maps so lines and background shades match
    cmap_lines = plt.get_cmap('Set1')   # Strong colors for lines
    cmap_bg = plt.get_cmap('Pastel1')   # Soft colors for backgrounds
    
    line_colors = {parent: cmap_lines(i % 9) for i, parent in enumerate(parent_names)}
    bg_colors = {parent: cmap_bg(i % 9) for i, parent in enumerate(parent_names)}

    # Process and Plot
    for record in SeqIO.parse(args.chimeras, "fasta"):
        chimera_id = record.id
        seq = record.seq
        print(f"Profiling {chimera_id}...")
        
        positions = []
        rel_scores = {parent: [] for parent in parent_names}
        
        for start in range(0, len(seq) - args.window, args.step):
            end = start + args.window
            midpoint = start + (args.window // 2)
            subseq = seq[start:end]
            
            positions.append(midpoint)
            
            raw_scores = {}
            for parent in parent_names:
                raw_scores[parent] = calculate_k_log_likelihood(subseq, p_models[parent]['matrix'], p_models[parent]['map'], args.order)
            
            avg_score = np.mean(list(raw_scores.values()))
            
            for parent in parent_names:
                rel_scores[parent].append(raw_scores[parent] - avg_score)

        plt.figure(figsize=(14, 7))
        
        # Plot the likelihood lines
        for parent in parent_names:
            plt.plot(positions, rel_scores[parent], label=f"{parent} Signal", color=line_colors[parent], alpha=0.9, linewidth=2.5)

        # Draw a baseline at 0
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.4)

        # Shade the background based on true source
        seen_bg_labels = set()
        if chimera_id in ground_truth:
            for frag in ground_truth[chimera_id]:
                # Only add the label to the legend the first time we see this parent's background
                bg_label = f"True Region: {frag['source']}" if frag['source'] not in seen_bg_labels else ""
                seen_bg_labels.add(frag['source'])
                
                plt.axvspan(frag['start'], frag['stop'], color=bg_colors.get(frag['source'], 'lightgray'), alpha=0.5, label=bg_label)

                if frag['start'] > 1:
                    plt.axvline(x=frag['start'], color='black', linestyle=':', alpha=0.5)

        plt.title(f"Relative Markov Profile ({args.order}-Order): {chimera_id} (Window={args.window}bp)")
        plt.xlabel("Position")
        plt.ylabel("Relative Log-Likelihood")
        
        # Move legend outside the plot so it doesn't cover data
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        # Save plot
        output_filename = os.path.join(args.outdir, f"{chimera_id}_relative_profile_k{args.order}.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_filename}")


    # GLOBAL ACCURACY FOR DIFFERENT K
    print("\n--- Global Accuracy by Markov Order ---")
    k_values_to_test = [1, 2, 3, 4]
    
    for test_k in k_values_to_test:
        # Build k-order models
        test_models = {}
        for record in SeqIO.parse(args.parents, "fasta"):
            matrix, mapping = get_k_order_transition_matrix(record.seq, test_k)
            test_models[record.id] = {'matrix': matrix, 'map': mapping}
            
        # Determine accuracy
        accuracy, total_windows = get_global_accuracy(args.chimeras, ground_truth, test_models, test_k, args.window, args.step)
        print(f"Order k={test_k}: {accuracy:.2f}% accuracy (Evaluated {total_windows} windows)")

if __name__ == "__main__":
    main()