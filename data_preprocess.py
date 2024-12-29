from Bio import SeqIO
import re
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical
import pandas as pd
#Valid characters in amino acids
valid_chars = 'ACDEFGHIKLMNPQRSTVWY'




def load_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

# Load sequences from different FASTA files
hsp20_sequences = load_fasta("data/raw/HSP20.fasta")
hsp40_sequences = load_fasta("data/raw/HSP40.fasta")
hsp60_sequences = load_fasta("data/raw/HSP60.fasta")
hsp70_sequences = load_fasta("data/raw/HSP70.fasta")
hsp90_sequences = load_fasta("data/raw/HSP90.fasta")
hsp100_sequences = load_fasta("data/raw/HSP100.fasta")
non_hsp_sequences = load_fasta("data/raw/NonHSP.fasta")

#Labels
hsp20_labels = ['HSP20'] * len(hsp20_sequences)  # Label 0 for HSP20
hsp40_labels = ['HSP40'] * len(hsp40_sequences)  # Label 1 for HSP40
hsp60_labels = ['HSP60'] * len(hsp60_sequences)  # Label 2 for HSP60
hsp70_labels = ['HSP70'] * len(hsp70_sequences)  # Label 3 for HSP70
hsp90_labels = ['HSP90'] * len(hsp90_sequences)  # Label 4 for HSP90
hsp100_labels = ['HSP100'] * len(hsp100_sequences)  # Label 5 for HSP100
non_hsp_labels = ['NON_HSP'] * len(non_hsp_sequences)  # Label -1 for Non HSP



sequences = hsp20_sequences + hsp40_sequences + hsp60_sequences + hsp70_sequences + hsp90_sequences + hsp100_sequences + non_hsp_sequences 
labels = hsp20_labels + hsp40_labels + hsp60_labels + hsp70_labels + hsp90_labels + hsp100_labels + non_hsp_labels


data = {'Sequence': sequences, 'Label': labels} 
print(len(sequences))
print(len(labels))
df = pd.DataFrame(data) 
# Write to CSV 
df.to_csv('data/processed/sequences_with_labels.csv', index=False)


def clean_sequence(seq, valid_chars):
    #Remove invalid characters which are not amino acids.
    cleaned_seq = ''.join([c for c in seq if c in valid_chars])
    return cleaned_seq

cleaned_sequences = [clean_sequence(seq, valid_chars) for seq in sequences]






def one_hot_encode_sequences(sequences):

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
    

    
    return one_hot_encoded_sequences[0]

