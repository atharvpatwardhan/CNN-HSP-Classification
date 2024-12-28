from Bio import SeqIO
import re
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

fasta_source_file = 'data/test/HSP20.fasta'

sequences_data = list(SeqIO.parse(fasta_source_file, 'fasta'))

#Valid characters in amino acids
valid_chars = 'ACDEFGHIKLMNPQRSTVWY'

def clean_sequence(seq, valid_chars):
    #Remove invalid characters which are not amino acids.
    cleaned_seq = ''.join([c for c in seq if c in valid_chars])
    return cleaned_seq




def one_hot_encode_sequences(sequences):
    cleaned_sequences = []
    for record in sequences:
        cleaned_seq = clean_sequence(str(record.seq), valid_chars)
        cleaned_sequences.append(cleaned_seq)



    #Convert sequences to integer indices to feed to one hot encoder
    char_index_dict = {char: i for i, char in enumerate(valid_chars)}
    integer_encoded_sequences = [
        [char_index_dict[char] for char in sequence] for sequence in cleaned_sequences
    ]
    encoder = OneHotEncoder(categories=[list(range(len(valid_chars)))])


    one_hot_encoded_sequences = []
    for sequence in integer_encoded_sequences:
        reshaped_sequence = np.array(sequence).reshape(-1, 1)
        encoded_sequence = encoder.fit_transform(reshaped_sequence)
        one_hot_encoded_sequences.append(encoded_sequence)

    return one_hot_encoded_sequences

