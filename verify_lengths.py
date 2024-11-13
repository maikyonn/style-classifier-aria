from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
from tqdm import tqdm
from datetime import datetime

# data_dir = "/project/jonmay_231/spangher/Projects/music-form-structure-modeling/aria-extended-gen-no-ft/synth_data"
data_dir = "datasets/17k-unstructured"
tokenizer = AbsTokenizer()

# Use process-safe counter
matching_count = Value('i', 0)
total_count = Value('i', 0)

def process_file(style_path, pbar, mismatch_file):
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
        with matching_count.get_lock():
            matching_count.value += 1
            pbar.set_description(f"Processing files (Matching: {matching_count.value})")
    else:
        # Write mismatched files to the log
        with open(mismatch_file, 'a') as f:
            f.write(f"File: {style_path}\n")
            f.write(f"MIDI file: {midi_path}\n")
            f.write(f"Text token count: {txt_token_count}\n")
            f.write(f"MIDI token count: {midi_token_count}\n")
            f.write("-" * 80 + "\n")

# Create timestamp for the mismatch file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
mismatch_file = f"mismatched_files_{timestamp}.txt"

# Write header to mismatch file
with open(mismatch_file, 'w') as f:
    f.write("Mismatched Files Log\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")

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
    futures = [executor.submit(process_file, path, pbar, mismatch_file) for path in style_paths]
    for _ in futures:
        _.result()
        pbar.update(1)

pbar.close()

# Add summary to mismatch file
with open(mismatch_file, 'a') as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write("Summary:\n")
    f.write(f"Files with matching token counts: {matching_count.value}/{total_count.value}\n")
    f.write(f"Percentage matching: {(matching_count.value/total_count.value)*100:.2f}%\n")

print(f"\nSummary:")
print(f"Files with matching token counts: {matching_count.value}/{total_count.value}")
print(f"Percentage matching: {(matching_count.value/total_count.value)*100:.2f}%")
print(f"Mismatched files have been saved to: {mismatch_file}")