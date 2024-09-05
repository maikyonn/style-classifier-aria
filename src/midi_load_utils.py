

import json
import os
import logging
import functools
from functools import partial
from multiprocessing import Pool
from typing import Callable, Iterable
from aria.tokenizer import Tokenizer
from aria.data.midi import MidiDict
from multiprocessing import Pool, get_start_method
from progress.bar import Bar
    
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
    _midi_dict = MidiDict.from_midi(path)
    seq = tokenizer.tokenize(_midi_dict)
    pure_seq = []
    for tok in seq:
        if tok[0] in ['piano', 'onset', 'dur']:
            pure_seq.extend(tokenizer.encode([tok]))
    
    return pure_seq

def get_style_sequence(path, tokenizer):
    with open(path, 'r') as file:
        content = file.read()
    
    # Tokenize the content into a list of characters
    tokens = list(content)
    encoded_tokens = tokenizer.encode(tokens)
    
    return encoded_tokens


def build_dataset(path, tokenizer):
    midi_sequences = []
    style_sequences = []

    # Define paths to midi and labels folders
    midi_folder = os.path.join(path, 'midi')
    labels_folder = os.path.join(path, 'labels')

    # Collect all midi and label files
    midi_files = sorted([os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith('.mid')])
    label_files = sorted([os.path.join(labels_folder, f) for f in os.listdir(labels_folder) if f.endswith('.txt')])

    # Ensure both lists have the same number of files
    if len(midi_files) != len(label_files):
        raise ValueError("Mismatch between number of MIDI and label files.")

    # Process each pair of midi and label files
    for midi_file, label_file in zip(midi_files, label_files):
        midi_seq = get_clean_midi_tokenize(midi_file, tokenizer)
        style_seq = get_style_sequence(label_file, tokenizer)

        # Verify that the MIDI sequence and style sequence are of the same length
        if len(midi_seq) != len(style_seq):
            raise ValueError(f"Length mismatch between MIDI file {os.path.basename(midi_file)} and label file {os.path.basename(label_file)}.")

        midi_sequences.append(midi_seq)
        style_sequences.append(style_seq)

    return midi_sequences, style_sequences

def chunk_sequences(sequences, max_len=1024, padding_value=0):
    chunked_sequences = []
    for seq in sequences:
        # Chunk the sequence into pieces of max_len
        for i in range(0, len(seq), max_len):
            chunk = seq[i:i + max_len]
            
            # If the chunk is shorter than max_len, pad it
            if len(chunk) < max_len:
                chunk += [padding_value] * (max_len - len(chunk))
                
            chunked_sequences.append(chunk)
    return chunked_sequences
