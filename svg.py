#!/usr/bin/env python
# coding: utf-8

# Part of Scaffold Generative Pretraining Project
# Author : Manas Mahale <manas.mahale@bcp.edu.in>

import re
import logging


from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger                                                                                                                                                               

from model.utils import set_seed
from model.model import GPT, GPTConfig
from model.utils import sample

import torch
from torch.utils.data import Dataset

# set up logging

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

# make deterministic
set_seed(42)


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = []
        for i in text:
            for j in set(i):
                chars.append(j)           
        chars = sorted(list(set(chars))) + ['<pad>']
        data_size, vocab_size = len(text), len(chars)
        print('Data has %d SMILES \n%d unique characters.' % (data_size, vocab_size))        
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + 1][0]
        dix = [self.stoi[s] for s in chunk] + [self.stoi['<pad>']] * (self.block_size - len(chunk))
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


block_size = 64

text = [i.strip() for i in open('./data/final/train.txt', 'r').readlines()]
train_dataset = CharDataset(text, block_size)


mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=2, n_head=2, n_embd=16)
model = GPT(mconf)

model.load_state_dict(torch.load('./ckpt/big_model.bin'))

RDLogger.DisableLog('rdApp.*')                                                                                                                                                           
print('\n**Generating Scaffold SMILES**\n')
valid = []
for n in range(1, 501):
    context = "C"
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...]
    y = sample(model, x, 25, temperature=1.0, sample=True, top_k=10)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])

    smiles = re.sub("<pad>","",completion)
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is not None:
        print(n, smiles)
        valid.append(smiles)
print('\n', len(valid)/5,'% Valid')

def plot_rdkit_svg_grid(mols, mols_per_row=2, filename="generated"):
    svg = Chem.Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, useSVG=True)
    if filename is not None:
        if not filename.endswith('.svg'):
            filename += '.svg'
        with open(filename, 'w') as f:
            f.write(svg)
    return svg 

plot_rdkit_svg_grid([Chem.MolFromSmiles(i) for i in valid])
