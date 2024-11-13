from aria.tokenizer import SeparatedAbsTokenizer
from aria.data.midi import MidiDict

with open("/project/jonmay_231/spangher/Projects/music-form-structure-modeling/aria-extended-gen-no-ft/synth_data/samples_957/2_style.txt", 'r') as file:
    content = file.read()
tokenizer = SeparatedAbsTokenizer()
tokens = list(content)
encoded_tokens = tokenizer.encode(tokens)
print(len(tokens))

_midi_dict = MidiDict.from_midi("/project/jonmay_231/spangher/Projects/music-form-structure-modeling/aria-extended-gen-no-ft/synth_data/samples_957/2_midi.mid")
seq = tokenizer.tokenize(_midi_dict)
pure_seq = []
for tok in seq:
    if tok[0] in ['piano', 'onset', 'dur']:
        pure_seq.extend(tokenizer.encode([tok]))
print(len(pure_seq))