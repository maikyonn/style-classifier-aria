from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
from tqdm import tqdm
import shutil

data_dir = "/project/jonmay_231/spangher/Projects/music-form-structure-modeling/aria-extended-gen-no-ft/synth_data"
output_dir = "datasets/17k-unstructured"
tokenizer = AbsTokenizer()

# Use process-safe counter
matching_count = Value('i', 0)
total_count = Value('i', 0)

def process_file(style_path, pbar):
    with total_count.get_lock():
        total_count.value += 1
    
    with open(style_path, 'r') as file:
        content = file.read()
    tokens = list(content)
    encoded_tokens = tokenizer.encode(tokens)
    txt_token_count = len(tokens)
    
    # Process corresponding MIDI file
    midi_filename = os.path.basename(style_path).replace("_style.txt", "_midi.mid")
    midi_path = os.path.join(os.path.dirname(style_path), midi_filename)
    _midi_dict = MidiDict.from_midi(midi_path)
    seq = tokenizer.tokenize(_midi_dict)
    pure_seq = []
    for tok in seq:
        if tok[0] in ['piano', 'onset', 'dur']:
            pure_seq.extend(tokenizer.encode([tok]))
    midi_token_count = len(pure_seq)
    
    if txt_token_count == midi_token_count:
        # Create relative path for maintaining directory structure
        rel_path = os.path.relpath(os.path.dirname(style_path), data_dir)
        new_dir = os.path.join(output_dir, rel_path)
        
        # Create destination directory if it doesn't exist
        os.makedirs(new_dir, exist_ok=True)
        
        # Copy both files to new location
        shutil.copy2(style_path, os.path.join(new_dir, os.path.basename(style_path)))
        shutil.copy2(midi_path, os.path.join(new_dir, os.path.basename(midi_path)))
        
        with matching_count.get_lock():
            matching_count.value += 1
            pbar.set_description(f"Processing files (Matching: {matching_count.value})")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Collect file paths
style_paths = []
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith("_style.txt"):
            style_path = os.path.join(root, filename)
            style_paths.append(style_path)

# Create progress bar
pbar = tqdm(total=len(style_paths), desc="Processing files (Matching: 0)", unit="file")

# Process files in parallel using thread pool
with ThreadPoolExecutor(max_workers=64) as executor:
    futures = [executor.submit(process_file, path, pbar) for path in style_paths]
    for _ in futures:
        _.result()
        pbar.update(1)

pbar.close()

print(f"\nSummary:")
print(f"Files with matching token counts: {matching_count.value}/{total_count.value}")
print(f"Percentage matching: {(matching_count.value/total_count.value)*100:.2f}%")
print(f"Matching files have been copied to: {output_dir}")