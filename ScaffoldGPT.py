#!/usr/bin/env python
# coding: utf-8

# Part of Scaffold Generative Pretraining Project
# Author : Manas Mahale <manas.mahale@bcp.edu.in>

import logging                                                                                                                                                        

from model.utils import set_seed
from model.model import GPT, GPTConfig
from model.trainer import Trainer, TrainerConfig
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
        for i in data:
            for j in set(i):
                chars.append(j)           
        chars = sorted(list(set(chars))) + ['<pad>']
        data_size, vocab_size = len(data), len(chars)
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
train_text = [i.strip() for i in open('./data/final/train.txt', 'r').readlines()]
test_text  = [i.strip() for i in open('./data/final/test.txt', 'r').readlines()]
train_dataset = CharDataset(train_text, block_size)
test_dataset  = CharDataset(test_text, block_size)


mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=12, n_head=12, n_embd=768)
model = GPT(mconf)


tconf = TrainerConfig(max_epochs=5, batch_size=8, learning_rate=1e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                      ckpt_path='./ckpt/big_model.bin')
trainer = Trainer(model, train_dataset, test_dataset, tconf)

trainer.train()
