import tqdm
import torch
from torch import nn
from torch import optim

from models import KBCModel, read_txt
from regularizers import Regularizer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True, finetune=False
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.finetune = finetune
        self.analogy_entids = [int(ent) for ent in read_txt('data/analogy/analogy_ent_id')]
        self.analogy_relids = [int(rel) for rel in read_txt('data/analogy/analogy_rel_id')]

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2] if not self.finetune else input_batch[:, 3]

                l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')