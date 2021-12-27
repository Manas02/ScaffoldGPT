#!/usr/bin/env python
# coding: utf-8
# Part of Scaffold Generative Pretraining Project
# Author : Manas Mahale <manas.mahale@bcp.edu.in>


from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

with open('data/train.txt', 'r') as f:
    smi = [i.strip() for i in f.readlines()]

print('***creating & writing to scaffolds***')

with open('data/train_scaffold.txt', 'w') as f:
    for i in tqdm(smi):
        f.write(MurckoScaffold.MurckoScaffoldSmilesFromSmiles(i, includeChirality=False) + '\n')
