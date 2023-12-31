import os
import torch
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
import re
import numpy as np
import gc
# tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
# model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
# Load model directly
if os.name == 'nt':
    prot_path = 'E:/OwnCode/PionicNet/code/tools/ProtT5'
else:
    prot_path = ''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(prot_path, do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained(prot_path).to(device)

# prepare your protein sequences as a list
sequence_examples = ["PRTEINO", "PRTEINO"]

# replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# generate embeddings
with torch.no_grad():
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

# extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]) 
emb_0 = embedding_repr.last_hidden_state[0,:7] # shape (7 x 1024)
# same for the second ([1,:]) sequence but taking into account different sequence lengths ([1,:8])
emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)
print(emb_0)
print(emb_1)
# # if you want to derive a single representation (per-protein embedding) for the whole protein
# emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)
