import torch
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple
from itertools import groupby
import argparse
from datetime import datetime

# Add these imports for the model and data components
from src.MidiStyleTransformer import MidiStyleTransformer
from src.MidiDataModule import MidiDataModule
from src.model import ModelConfig
from src.MidiStyleDataset import MidiStyleDataset

# Optional: Set CUDA optimization for consistency with training
torch.set_float32_matmul_precision('high')

def get_latest_run(runs_dir: str = 'runs') -> Path:
    """Get the path to the latest run directory."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs directory '{runs_dir}' not found")
    
    runs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not runs:
        raise FileNotFoundError(f"No run directories found in '{runs_dir}'")
    
    def parse_date(run_path: Path) -> datetime:
        # Extract the datetime part after 'run_'
        date_str = run_path.name[4:]  # Remove 'run_' prefix
        
        try:
            # Try format: YYYYMMDD_HHMMSS
            return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        except ValueError:
            try:
                # Try format: YYYYMMDD
                return datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                print(f"Warning: Could not parse date from directory: {run_path.name}")
                return datetime.min

    # Sort runs by date and get the latest
    sorted_runs = sorted(runs, key=parse_date, reverse=True)
    latest_run = sorted_runs[0]
    
    # Print all found runs for debugging
    print("\nFound runs (sorted by date):")
    for run in sorted_runs:
        print(f"- {run.name} ({parse_date(run)})")
    
    print(f"\nSelected latest run: {latest_run}")
    return latest_run

def get_best_checkpoint(run_dir: Path) -> Path:
    """Get the path to the best checkpoint in the run directory."""
    checkpoint_dir = run_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found in {run_dir}")
    
    checkpoints = list(checkpoint_dir.glob('*.ckpt'))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    def extract_val_loss(checkpoint_path):
        try:
            # Try format: midi-epoch-val_loss=X.XX.ckpt
            if 'val_loss=' in checkpoint_path.stem:
                return float(checkpoint_path.stem.split('val_loss=')[1])
            # Try format: midi-epoch-X.XX.ckpt
            return float(checkpoint_path.stem.split('-')[-1])
        except (ValueError, IndexError):
            return float('inf')
    
    best_checkpoint = min(checkpoints, key=extract_val_loss)
    
    if extract_val_loss(best_checkpoint) == float('inf'):
        last_checkpoint = checkpoint_dir / 'last.ckpt'
        if last_checkpoint.exists():
            print(f"Using last checkpoint: {last_checkpoint}")
            return last_checkpoint
        raise FileNotFoundError("Could not find a valid checkpoint")
    
    print(f"Using best checkpoint: {best_checkpoint}")
    return best_checkpoint

class InferenceAnalyzer:
    def __init__(
        self,
        model: MidiStyleTransformer,
        tokenizer,
        run_path: str,
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.run_path = Path(run_path)
        self.device = device
        self.results_dir = self.run_path / 'inference'
        self.results_dir.mkdir(exist_ok=True)
        print(f"\nResults will be saved in: {self.results_dir}")

    @staticmethod
    def condense_sequence(sequence: List[str]) -> str:
        """Condense repeated tokens into count format."""
        if not sequence:
            return ""
        return ", ".join(f"{len(list(group))}x{char}" 
                        for char, group in groupby(sequence))

    def run_inference(self, dataloader) -> Tuple[float, float, List[dict]]:
        """Run inference and return metrics."""
        self.model.eval()
        results = []
        total_loss = 0
        all_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running inference"):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(inputs)
                logits = logits.transpose(1, 2)
                loss = torch.nn.functional.cross_entropy(
                    logits, labels, 
                    ignore_index=self.tokenizer.pad_id
                )
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Calculate metrics
                correct = (predictions == labels).sum().item()
                total = labels.ne(self.tokenizer.pad_id).sum().item()
                all_correct += correct
                total_tokens += total
                total_loss += loss.item()

                # Store batch results
                for pred, label in zip(predictions, labels):
                    pred_tokens = self.tokenizer.decode(pred.cpu().numpy())
                    label_tokens = self.tokenizer.decode(label.cpu().numpy())
                    results.append({
                        'prediction': pred_tokens,
                        'label': label_tokens,
                        'condensed_prediction': self.condense_sequence(pred_tokens),
                        'condensed_label': self.condense_sequence(label_tokens),
                    })

        avg_loss = total_loss / len(dataloader)
        accuracy = all_correct / total_tokens
        return avg_loss, accuracy, results

    def save_results(self, results: List[dict], metrics: dict):
        """Save inference results and metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results with timestamp
        results_path = self.results_dir / f'results_{timestamp}.csv'
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to: {results_path}")
        
        # Save summary with timestamp
        summary_path = self.results_dir / f'summary_{timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write("Inference Results Summary:\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Samples: {len(results)}\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nSample Predictions (first 10):\n")
            for i, result in enumerate(results[:10], 1):
                f.write(f"\nSample {i}:\n")
                f.write(f"Prediction: {result['condensed_prediction']}\n")
                f.write(f"Label: {result['condensed_label']}\n")
        print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on trained model')
    parser.add_argument('--run_path', type=str, default=None,
                       help='Path to specific run directory (default: use latest)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint (default: use best)')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--runs_dir', type=str, default='runs',
                       help='Directory containing all runs')
    args = parser.parse_args()

    # Get run directory and checkpoint
    run_path = Path(args.run_path) if args.run_path else get_latest_run(args.runs_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_best_checkpoint(run_path)

    # Load model and data
    print("\nLoading model from checkpoint...")
    model = MidiStyleTransformer.load_from_checkpoint(str(checkpoint_path))
    model.to('cuda')
    model.eval()
    
    print("Setting up data module...")
    datamodule = MidiDataModule(args.dataset_dir)
    datamodule.setup()

    # Initialize analyzer
    analyzer = InferenceAnalyzer(
        model=model,
        tokenizer=datamodule.tokenizer,
        run_path=run_path
    )

    # Run inference
    print("\nRunning inference...")
    loss, accuracy, results = analyzer.run_inference(datamodule.val_dataloader())

    # Save results
    print("\nSaving results...")
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    analyzer.save_results(results, metrics)
    
    # Print final metrics
    print(f"\nFinal Metrics:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()