from Bio import SeqIO
import re
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

fasta_source_file = 'data/test/HSP20.fasta'

sequences = list(SeqIO.parse(fasta_source_file, 'fasta'))

def clean_sequence(seq, valid_chars):
    # Remove invalid characters
    cleaned_seq = ''.join([c for c in seq if c in valid_chars])
    return cleaned_seq

#Valid characters in amino acids
valid_chars = 'ACDEFGHIKLMNPQRSTVWY'

cleaned_sequences = []
for record in sequences:
    cleaned_seq = clean_sequence(str(record.seq), valid_chars)
    cleaned_sequences.append(cleaned_seq)


char_index_dict = {char: i for i, char in enumerate(valid_chars)}

# Convert sequences to integer indices
integer_encoded_sequences = [
    [char_index_dict[char] for char in sequence] for sequence in cleaned_sequences
]
encoder = OneHotEncoder(categories=[list(range(len(valid_chars)))])


one_hot_encoded_sequences = []
for sequence in integer_encoded_sequences:
    # Reshape to (n, 1) for sklearn compatibility
    reshaped_sequence = np.array(sequence).reshape(-1, 1)
    encoded_sequence = encoder.fit_transform(reshaped_sequence)
    one_hot_encoded_sequences.append(encoded_sequence)

print("One-hot encoded shape:", one_hot_encoded_sequences[0].shape)
print("One-hot encoded example:", one_hot_encoded_sequences[0])
