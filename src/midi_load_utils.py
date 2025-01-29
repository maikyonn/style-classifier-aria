import json
import os
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Callable, Iterable
from tqdm.auto import tqdm
from aria.tokenizer import Tokenizer
from aria.data.midi import MidiDict

def setup_logger():
    # Get logger and reset all handlers
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_seqs(
    tokenizer: Tokenizer,
    midi_dict_iter: Iterable,
):
    num_proc = os.cpu_count()

    # Can't pickle geneator object when start method is spawn
    if get_start_method() == "spawn":
        logging.info(
            "Converting generator to list due to multiprocessing start method"
        )
        midi_dict_iter = [_ for _ in midi_dict_iter]

    with Pool(16) as pool:
        results = pool.imap(
            functools.partial(_get_seqs, _tokenizer=tokenizer), midi_dict_iter
        )

        yield from results

def _get_seqs(_entry: MidiDict | dict, _tokenizer: Tokenizer):
    logger = setup_logger()

    if isinstance(_entry, str):
        _midi_dict = MidiDict.from_msg_dict(json.loads(_entry.rstrip()))
    elif isinstance(_entry, dict):
        _midi_dict = MidiDict.from_msg_dict(_entry)
    elif isinstance(_entry, MidiDict):
        _midi_dict = _entry
    else:
        raise Exception

    try:
        _tokenized_seq = _tokenizer.tokenize(_midi_dict)
    except Exception as e:
        print(e)
        logger.info(f"Skipping midi_dict: {e}")
        return
    else:
        if _tokenizer.unk_tok in _tokenized_seq:
            logger.warning("Unknown token seen while tokenizing midi_dict")
        return _tokenized_seq

def get_combined_slices(data, slice_length, overlap=512):
    slices = []
    step = slice_length - overlap
    for start_idx in range(0, len(data), step):
        end_idx = start_idx + slice_length
        slice = data[start_idx:end_idx]
        if len(slice) == slice_length:
            slices.append(slice)
    return slices


def _format(tok):
    # This is required because json formats tuples into lists
    if isinstance(tok, list):
        return tuple(tok)
    return tok

def process_line(line, base_path, overlap_token_amount, tokenizer, seq_len):
    quant_midi_path, perf_midi_path = line.strip().split('|')

    quant_midi_dict = MidiDict.from_midi(os.path.join(base_path, quant_midi_path))
    perf_midi_dict = MidiDict.from_midi(os.path.join(base_path, perf_midi_path))
    tokenized_quant_midi = tokenizer._tokenize_midi_dict(midi_dict=quant_midi_dict)
    tokenized_perf_midi = tokenizer._tokenize_midi_dict(midi_dict=perf_midi_dict)

    tuple_quant_tokenized_midi = [_format(tok) for tok in tokenized_quant_midi]
    tuple_perf_tokenized_midi = [_format(tok) for tok in tokenized_perf_midi]

    quant_combined_slices = get_combined_slices(tuple_quant_tokenized_midi, seq_len, overlap=128)
    perf_combined_slices = get_combined_slices(tuple_perf_tokenized_midi, seq_len, overlap=128)
    
    return quant_combined_slices, perf_combined_slices


def load_midi_and_tokenize_multi(path, tokenizer, max_seq_len, overlap_token_amount=128, limit=10000, num_workers=30):
    x = []
    y = []
    with open(path) as datafile:
        lines = datafile.readlines()
        lines = lines[:limit]
        
        with Pool(num_workers) as pool:
            process_func = partial(process_line, base_path=path, overlap_token_amount=overlap_token_amount, tokenizer=tokenizer, seq_len=max_seq_len)
            
            bar = Bar('Processing', max=len(lines))
            for quant_combined_slices, perf_combined_slices in pool.imap(process_func, lines):
                x.extend(quant_combined_slices)
                y.extend(perf_combined_slices)
                bar.next()
            bar.finish()
    
    return x, y


def get_clean_midi_tokenize(path, tokenizer):
    try:
        _midi_dict = MidiDict.from_midi(path)
        seq = tokenizer.tokenize(_midi_dict)
        pure_seq = []
        for tok in seq:
            if tok[0] in ['piano', 'onset', 'dur']:
                pure_seq.extend(tokenizer.encode([tok]))
        return pure_seq
    except Exception as e:
        raise Exception(f"Error processing {path}: {str(e)}")

def get_style_sequence(path, tokenizer):
    try:
        with open(path, 'r') as file:
            content = file.read()
        tokens = list(content)
        encoded_tokens = tokenizer.encode(tokens)
        return encoded_tokens
    except Exception as e:
        raise Exception(f"Error processing {path}: {str(e)}")

# src/midi_load_utils.py

def process_file_pair(args, tokenizer):
    """
    Process a pair of MIDI and label files.
    Returns (midi_seq, style_seq) or None if error/mismatch.
    """
    midi_file, label_file = args
    try:
        midi_seq = get_clean_midi_tokenize(midi_file, tokenizer)
        style_seq = get_style_sequence(label_file, tokenizer)
        
        if len(midi_seq) != len(style_seq):
            return None
        return midi_seq, style_seq
    except Exception as e:
        return None


def build_dataset(path, tokenizer):
    midi_sequences = []
    style_sequences = []
    logger = setup_logger()

    # Define paths to midi and labels folders
    midi_folder = os.path.join(path, 'midi')
    labels_folder = os.path.join(path, 'labels')

    # Collect all midi and label files
    midi_files = sorted([os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith('.mid')])
    label_files = sorted([os.path.join(labels_folder, f) for f in os.listdir(labels_folder) if f.endswith('.txt')])

    if len(midi_files) != len(label_files):
        raise ValueError("Mismatch between number of MIDI and label files.")

    # Calculate optimal number of workers
    num_workers = min(cpu_count(), 16)  # Cap at 16 workers
    
    print(f"\nProcessing {len(midi_files)} files using {num_workers} workers...")
    
    # Process files in parallel with progress bar
    with Pool(num_workers) as pool:
        # Use partial to pass the tokenizer to process_file_pair
        process_func = partial(process_file_pair, tokenizer=tokenizer)
        results = list(tqdm(
            pool.imap(process_func, zip(midi_files, label_files)),
            total=len(midi_files),
            desc="Processing MIDI and label files",
            unit="files"
        ))

    # Filter out None results and separate sequences
    valid_results = [r for r in results if r is not None]
    if valid_results:
        midi_sequences, style_sequences = zip(*valid_results)
    
    print(f"\nSuccessfully processed {len(midi_sequences)} file pairs")
    print(f"Skipped {len(midi_files) - len(midi_sequences)} files due to errors or length mismatches")

    return list(midi_sequences), list(style_sequences)

def chunk_sequences(sequences, max_len=1024, padding_value=0, stride=512):
    """
    Chunk sequences into fixed-size windows with padding using sliding window.
    Uses overlapping windows with specified stride (default: 32).
    
    Args:
        sequences: List of sequences to chunk
        max_len: Maximum length of each chunk
        padding_value: Value to use for padding
        stride: Number of tokens to shift window by (default: 32)
    """
    chunked_sequences = []
    
    # Process sequences with progress bar
    for seq in tqdm(sequences, desc="Chunking sequences", unit="sequence"):
        # Create sliding windows with specified stride
        for start_idx in range(0, len(seq), stride):  # Move window by stride tokens
            # Get chunk of size max_len
            chunk = seq[start_idx:start_idx + max_len]
            
            # Always pad to max_len
            if len(chunk) < max_len:
                chunk = chunk + [padding_value] * (max_len - len(chunk))
            
            chunked_sequences.append(chunk)
    
    print(f"Created {len(chunked_sequences)} chunks (stride={stride}) from {len(sequences)} sequences")
    return chunked_sequences

def prepare_midi_for_inference(midi_path, max_len=512, tokenizer=None):
    """
    Load, tokenize and chunk a single MIDI file for inference using a sliding window.
    
    Args:
        midi_path: Path to the MIDI file
        max_len: Maximum length of each chunk (default: 512)
        tokenizer: The AbsTokenizer instance
        
    Returns:
        list: List of chunked sequences ready for inference
        int: Stride size (1 for sliding window)
    """
    try:
        # Get the MIDI tokens using the same tokenization as training
        midi_seq = get_clean_midi_tokenize(midi_path, tokenizer)
        
        # Create chunks with sliding window (stride=1)
        chunks = []
        stride = 1  # Shift by 1 token at a time
        
        # For each possible window start position
        for start_idx in range(0, len(midi_seq)):
            # Get chunk of size max_len, pad if needed
            chunk = midi_seq[start_idx:start_idx + max_len]
            if len(chunk) < max_len:
                chunk = chunk + [tokenizer.encode(["<P>"])[0]] * (max_len - len(chunk))
            chunks.append(chunk)
            
        print(f"Created {len(chunks)} chunks using sliding window from MIDI file: {midi_path}")
        return chunks, stride
        
    except Exception as e:
        raise Exception(f"Error processing {midi_path}: {str(e)}")

