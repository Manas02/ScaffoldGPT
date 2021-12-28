#!/usr/bin/env python
# coding: utf-8
# Part of Scaffold Generative Pretraining Project
# Author : Manas Mahale <manas.mahale@bcp.edu.in>


from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

print('***Reading Data***')
with open('./data/raw/train.txt', 'r') as f:
    smi = [i.strip() for i in f.readlines()]

print('\n\tTrain Dataset\n***Creating & Writing scaffolds***')

with open('./data/final/train_scaffold.txt', 'w') as f:
    for i in tqdm(smi[:int(len(smi)*0.9)]):
        f.write(MurckoScaffold.MurckoScaffoldSmilesFromSmiles(i, includeChirality=False) + '\n')


print('\n\tTest Dataset\n***Creating & Writing scaffolds***')

with open('./data/final/test_scaffold.txt', 'w') as f:
    for i in tqdm(smi[int(len(smi)*0.9):]):
        f.write(MurckoScaffold.MurckoScaffoldSmilesFromSmiles(i, includeChirality=False) + '\n')
