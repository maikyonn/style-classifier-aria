import os
import torch
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from itertools import groupby
import random

from train_bert import MidiClassifier
from src.midi_load_utils import prepare_midi_for_inference
from aria.tokenizer import AbsTokenizer

# Set CUDA optimization for consistency with training
torch.set_float32_matmul_precision('medium')

def get_latest_checkpoint(checkpoints_dir: str = 'checkpoints') -> Path:
    """Get the path to the latest checkpoint."""
    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Checkpoints directory '{checkpoints_dir}' not found")
    
    checkpoints = [f for f in checkpoints_path.iterdir() if f.is_file() and f.suffix == '.ckpt']
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in '{checkpoints_dir}'")
    
    # Sort checkpoints by modification time
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    print(f"\nUsing latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def get_file_pairs(input_dir: str, limit: int = None) -> list:
    """Get pairs of MIDI and style files from the input directory."""
    midi_dir = os.path.join(input_dir, 'midi')
    labels_dir = os.path.join(input_dir, 'labels')
    
    print(f"\nLooking for files in:")
    print(f"MIDI directory: {midi_dir}")
    print(f"Labels directory: {labels_dir}")
    
    if not os.path.exists(midi_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Both 'midi' and 'labels' directories must exist in {input_dir}")
    
    # Get all files and sort them using Path objects
    midi_files = {Path(f.path).stem.split('_midi')[0]: f.path 
                 for f in os.scandir(midi_dir) 
                 if f.name.endswith('.mid')}
    style_files = {Path(f.path).stem.split('_style')[0]: f.path 
                  for f in os.scandir(labels_dir) 
                  if f.name.endswith('.txt')}
    
    print(f"\nFound {len(midi_files)} MIDI files and {len(style_files)} style files")
    
    # Find matching pairs
    pairs = []
    for stem in sorted(midi_files.keys()):
        if stem in style_files:
            midi_path = midi_files[stem]
            style_path = style_files[stem]
            pairs.append((midi_path, style_path))
    
    if not pairs:
        print("\nWARNING: No matching pairs found!")
        print("First few MIDI files:", list(midi_files.keys())[:5])
        print("First few style files:", list(style_files.keys())[:5])
        return pairs
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Apply limit if specified
    if limit is not None and pairs:
        original_length = len(pairs)
        pairs = pairs[:limit]
        print(f"\nRandomly selected {limit} pairs from {original_length} total pairs")
        
        # Print some sample pairs
        print("\nSample of selected pairs:")
        for midi_path, style_path in pairs[:5]:  # Show first 5 pairs
            print(f"Matched: {os.path.basename(midi_path)} <-> {os.path.basename(style_path)}")
        if len(pairs) > 5:
            print("...")
    
    print(f"\nTotal pairs to process: {len(pairs)}")
    return pairs

class BatchInferenceAnalyzer:
    def __init__(
        self,
        model: MidiClassifier,
        output_dir: str = 'batch_inference_results',
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        self.results_dir = Path(output_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AbsTokenizer()
        print("Loading tokenizer and adding tokens...")
        self.tokenizer.add_tokens_to_vocab(["A", "B", "C", "D"])
        self.pad_token = self.tokenizer.encode(["<P>"])[0]
        
        print(f"\nResults will be saved in: {self.results_dir}")

    @staticmethod
    def condense_sequence(sequence):
        """Condense repeated tokens into count format."""
        if not sequence:
            return ""
        return ", ".join(f"{len(list(group))}x{item}" 
                        for item, group in groupby(sequence))

    def predict_midi(self, midi_path: str, style_path: str = None) -> dict:
        """Run inference on a single MIDI file."""
        self.model.eval()
        
        # Prepare MIDI data using tokenizer
        chunked_sequences, stride = prepare_midi_for_inference(
            midi_path,
            max_len=self.model.max_length,
            tokenizer=self.tokenizer
        )
        
        # Initialize predictions storage
        true_labels = None
        if style_path:
            with open(style_path, 'r') as f:
                true_style_sequence = f.read().strip()
            true_labels = list(true_style_sequence)
            predictions_per_position = [[] for _ in range(len(true_labels))]
        else:
            predictions_per_position = None
        
        # Process each chunk
        for i, chunk in enumerate(chunked_sequences):
            chunk_tensor = torch.tensor(chunk).unsqueeze(0)
            attention_mask = (chunk_tensor != self.pad_token).long()
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    chunk_tensor = chunk_tensor.cuda()
                    attention_mask = attention_mask.cuda()
                
                logits = self.model(chunk_tensor, attention_mask)
                predictions = torch.argmax(logits.squeeze(0), dim=-1)
                predictions = predictions.cpu().numpy()
                
                # Initialize predictions_per_position if not done yet
                if predictions_per_position is None:
                    predictions_per_position = [[] for _ in range(len(predictions))]
                
                # Add predictions to their corresponding positions
                for pos, pred in enumerate(predictions):
                    if i + pos < len(predictions_per_position):
                        predictions_per_position[i + pos].append(pred)
        
        # Aggregate predictions for each position (majority vote)
        all_predictions = []
        for pos_predictions in predictions_per_position:
            if pos_predictions:  # Check if we have any predictions for this position
                pred = max(set(pos_predictions), key=pos_predictions.count)
                all_predictions.append(pred)
        
        # Convert numeric predictions to style labels
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        predicted_styles = [label_map[pred] for pred in all_predictions]
        
        # Calculate accuracy if true labels are available
        accuracy = None
        if true_labels:
            min_len = min(len(predicted_styles), len(true_labels))
            predicted_styles = predicted_styles[:min_len]
            true_labels = true_labels[:min_len]
            correct = sum(p == t for p, t in zip(predicted_styles, true_labels))
            accuracy = correct / len(true_labels)
        
        # Get condensed sequences
        condensed_pred = self.condense_sequence(predicted_styles)
        condensed_true = self.condense_sequence(true_labels) if true_labels else None
        
        return {
            'predictions': predicted_styles,
            'condensed_predictions': condensed_pred,
            'true_sequence': condensed_true,
            'accuracy': accuracy
        }

def main():
    parser = argparse.ArgumentParser(description='Run batch inference on MIDI files')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing midi and labels subdirectories')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint (default: use latest)')
    parser.add_argument('--output_dir', type=str, default='batch_inference_results',
                       help='Directory to save results')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum number of file pairs to process (default: 1000)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get checkpoint
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_latest_checkpoint()

    # Load model
    print("\nLoading model from checkpoint...")
    model = MidiClassifier.load_from_checkpoint(str(checkpoint_path))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Initialize analyzer
    analyzer = BatchInferenceAnalyzer(
        model=model,
        output_dir=args.output_dir
    )

    # Get file pairs with limit
    file_pairs = get_file_pairs(args.input_dir, args.limit)
    
    # Prepare results storage
    results_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f'batch_results_{timestamp}.csv'
    
    # Process each pair with progress bar
    print("\nProcessing file pairs...")
    for midi_path, style_path in tqdm(file_pairs, desc="Processing files"):
        try:
            # Run inference
            result = analyzer.predict_midi(midi_path, style_path)
            
            # Store results
            results_data.append({
                'midi_file': os.path.basename(midi_path),
                'style_file': os.path.basename(style_path),
                'predicted_sequence': result['condensed_predictions'],
                'true_sequence': result['true_sequence'],
                'accuracy': result['accuracy'],
                'status': 'success'
            })
            
        except Exception as e:
            print(f"\nError processing {midi_path}: {str(e)}")
            results_data.append({
                'midi_file': os.path.basename(midi_path),
                'style_file': os.path.basename(style_path),
                'predicted_sequence': None,
                'true_sequence': None,
                'accuracy': None,
                'status': f'error: {str(e)}'
            })
    
    # Save results to CSV
    df = pd.DataFrame(results_data)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary statistics
    successful_files = df[df['status'] == 'success']
    if not successful_files.empty:
        mean_accuracy = successful_files['accuracy'].mean()
        print(f"\nSummary Statistics:")
        print(f"Total files processed: {len(df)}")
        print(f"Successful predictions: {len(successful_files)}")
        print(f"Failed predictions: {len(df) - len(successful_files)}")
        print(f"Mean accuracy: {mean_accuracy:.4f}")

if __name__ == "__main__":
    main()