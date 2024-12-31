import torch
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple
from itertools import groupby
import argparse
from datetime import datetime

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

class InferenceAnalyzer:
    def __init__(
        self,
        model: MidiClassifier,
        output_dir: str = 'inference_results',
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
    def condense_sequence(sequence: List[str]) -> str:
        """Condense repeated tokens into count format."""
        if not sequence:
            return ""
        return ", ".join(f"{len(list(group))}x{char}" 
                        for char, group in groupby(sequence))

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
        for i, chunk in enumerate(tqdm(chunked_sequences, desc="Processing chunks")):
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

    def save_results(self, midi_path: str, results: dict):
        """Save inference results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f'results_{Path(midi_path).stem}_{timestamp}.txt'
        
        with open(results_path, 'w') as f:
            f.write("MIDI Style Inference Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input MIDI: {midi_path}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write(f"Predicted Sequence: {results['condensed_predictions']}\n")
            if results['true_sequence'] is not None:
                f.write(f"True Sequence: {results['true_sequence']}\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            
            f.write("\nRaw Predictions:\n")
            f.write("".join(results['predictions']))
        
        print(f"\nResults saved to: {results_path}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on MIDI file using trained model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint (default: use latest)')
    parser.add_argument('--midi_path', type=str, required=True,
                       help='Path to input MIDI file')
    parser.add_argument('--style_path', type=str,
                       help='Path to true style file (optional)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save results')
    args = parser.parse_args()

    # Get checkpoint
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_latest_checkpoint()

    # Load model
    print("\nLoading model from checkpoint...")
    model = MidiClassifier.load_from_checkpoint(str(checkpoint_path))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Initialize analyzer
    analyzer = InferenceAnalyzer(
        model=model,
        output_dir=args.output_dir
    )

    # Run inference
    print("\nRunning inference...")
    results = analyzer.predict_midi(args.midi_path, args.style_path)

    # Save results
    analyzer.save_results(args.midi_path, results)
    
    # Print results
    print("\nResults:")
    print(f"Predicted Sequence: {results['condensed_predictions']}")
    if results['accuracy'] is not None:
        print(f"True Sequence: {results['true_sequence']}")
        print(f"Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()