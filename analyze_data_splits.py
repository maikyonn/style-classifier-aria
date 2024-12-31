import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import pandas as pd

from src.MidiDataModule import MidiDataModule
from src.MidiStyleDataset import MidiStyleDataset

class DatasetAnalyzer:
    def __init__(self, data_dir: str, batch_size: int = 32, max_len: int = 512):
        """Initialize the analyzer with the same parameters as your training."""
        self.data_module = MidiDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            max_len=max_len
        )
        self.data_module.setup()
        
        # Get the full dataset before splitting
        self.full_dataset = self.data_module.train_dataset.dataset  # Get the original dataset
        self.train_indices = self.data_module.train_dataset.indices
        self.val_indices = self.data_module.val_dataset.indices

    def analyze_label_distribution(self) -> Dict:
        """Analyze the distribution of labels in train and validation sets."""
        train_labels = defaultdict(int)
        val_labels = defaultdict(int)
        
        print("\nAnalyzing label distribution...")
        
        # Count labels in training set
        for idx in tqdm(self.train_indices, desc="Processing training set"):
            _, style_labels = self.full_dataset[idx]
            unique_labels = torch.unique(style_labels[style_labels != -100])
            for label in unique_labels:
                train_labels[label.item()] += 1
                
        # Count labels in validation set
        for idx in tqdm(self.val_indices, desc="Processing validation set"):
            _, style_labels = self.full_dataset[idx]
            unique_labels = torch.unique(style_labels[style_labels != -100])
            for label in unique_labels:
                val_labels[label.item()] += 1
        
        return {
            'train': dict(train_labels),
            'val': dict(val_labels)
        }

    def compute_sequence_similarity(self, n_samples: int = 1000) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Compute similarity between sequences in training and validation sets.
        Uses random sampling for large datasets.
        """
        print("\nComputing sequence similarities...")
        
        # Randomly sample indices if dataset is too large
        train_samples = np.random.choice(self.train_indices, min(n_samples, len(self.train_indices)), replace=False)
        val_samples = np.random.choice(self.val_indices, min(n_samples, len(self.val_indices)), replace=False)
        
        # Get sequences
        train_sequences = []
        val_sequences = []
        
        print("Loading training sequences...")
        for idx in tqdm(train_samples):
            midi_seq, _ = self.full_dataset[idx]
            train_sequences.append(midi_seq.numpy())
            
        print("Loading validation sequences...")
        for idx in tqdm(val_samples):
            midi_seq, _ = self.full_dataset[idx]
            val_sequences.append(midi_seq.numpy())
            
        # Convert to numpy arrays
        train_sequences = np.array(train_sequences)
        val_sequences = np.array(val_sequences)
        
        # Reshape sequences to 2D
        train_2d = train_sequences.reshape(len(train_sequences), -1)
        val_2d = val_sequences.reshape(len(val_sequences), -1)
        
        print("Computing cosine similarity...")
        # Compute cosine similarity between all pairs
        similarity_matrix = cosine_similarity(train_2d, val_2d)
        
        # Find highly similar pairs
        similar_pairs = []
        threshold = 0.95  # Adjust this threshold as needed
        
        for i in range(len(train_samples)):
            for j in range(len(val_samples)):
                if similarity_matrix[i, j] > threshold:
                    similar_pairs.append((train_samples[i], val_samples[j]))
        
        return similarity_matrix, similar_pairs

    def plot_similarity_heatmap(self, similarity_matrix: np.ndarray, output_path: str = 'similarity_heatmap.png'):
        """Plot a heatmap of sequence similarities."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='YlOrRd')
        plt.title('Sequence Similarity Between Train and Validation Sets')
        plt.xlabel('Validation Sequences')
        plt.ylabel('Training Sequences')
        plt.savefig(output_path)
        plt.close()

    def plot_label_distribution(self, label_dist: Dict, output_path: str = 'label_distribution.png'):
        """Plot the distribution of labels in train and validation sets."""
        labels = sorted(set(list(label_dist['train'].keys()) + list(label_dist['val'].keys())))
        
        train_counts = [label_dist['train'].get(label, 0) for label in labels]
        val_counts = [label_dist['val'].get(label, 0) for label in labels]
        
        # Normalize counts to percentages
        train_total = sum(train_counts)
        val_total = sum(val_counts)
        train_pcts = [count/train_total*100 for count in train_counts]
        val_pcts = [count/val_total*100 for count in val_counts]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        ax1.bar(x - width/2, train_counts, width, label='Train')
        ax1.bar(x + width/2, val_counts, width, label='Validation')
        ax1.set_ylabel('Count')
        ax1.set_title('Label Distribution (Raw Counts)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        
        # Percentages
        ax2.bar(x - width/2, train_pcts, width, label='Train')
        ax2.bar(x + width/2, val_pcts, width, label='Validation')
        ax2.set_ylabel('Percentage')
        ax2.set_title('Label Distribution (Percentages)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze dataset splits for potential leakage')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--n_samples', type=int, default=1000, 
                       help='Number of samples to use for similarity analysis')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path('dataset_analysis')
    output_dir.mkdir(exist_ok=True)

    # Initialize analyzer
    analyzer = DatasetAnalyzer(args.data_dir)

    # Analyze label distribution
    label_dist = analyzer.analyze_label_distribution()
    
    # Print label distribution statistics
    print("\nLabel Distribution Statistics:")
    print("\nTraining Set:")
    for label, count in label_dist['train'].items():
        print(f"Label {label}: {count}")
    print("\nValidation Set:")
    for label, count in label_dist['val'].items():
        print(f"Label {label}: {count}")

    # Plot label distribution
    analyzer.plot_label_distribution(label_dist, str(output_dir / 'label_distribution.png'))

    # Compute and analyze sequence similarities
    similarity_matrix, similar_pairs = analyzer.compute_sequence_similarity(args.n_samples)
    
    # Plot similarity heatmap
    analyzer.plot_similarity_heatmap(similarity_matrix, str(output_dir / 'similarity_heatmap.png'))

    # Report findings
    print("\nAnalysis Results:")
    print(f"Number of highly similar pairs found: {len(similar_pairs)}")
    if similar_pairs:
        print("\nSample of similar pairs (train_idx, val_idx):")
        for pair in similar_pairs[:10]:  # Show first 10 pairs
            print(pair)

    # Save detailed results
    results = {
        'label_distribution': label_dist,
        'similar_pairs': similar_pairs,
        'similarity_stats': {
            'mean': float(np.mean(similarity_matrix)),
            'median': float(np.median(similarity_matrix)),
            'max': float(np.max(similarity_matrix)),
            'min': float(np.min(similarity_matrix)),
            'std': float(np.std(similarity_matrix))
        }
    }

    # Save results to CSV
    df = pd.DataFrame(results['similarity_stats'], index=[0])
    df.to_csv(output_dir / 'similarity_stats.csv', index=False)

    # Save similar pairs
    if similar_pairs:
        pairs_df = pd.DataFrame(similar_pairs, columns=['train_idx', 'val_idx'])
        pairs_df.to_csv(output_dir / 'similar_pairs.csv', index=False)

if __name__ == "__main__":
    main() 