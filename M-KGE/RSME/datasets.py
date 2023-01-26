from pathlib import Path
import pkg_resources
import pickle
from typing import Dict, Tuple, List
import tqdm
import numpy as np
import torch
from models import KBCModel, read_txt


DATA_PATH=Path('data')

class Dataset(object):
    def __init__(self, name: str):
        self.root = DATA_PATH / name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)

        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 5, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        mean_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, mean_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
    
    
class AnalogyDataset(object):
    def __init__(self, name: str):
        self.root = DATA_PATH / name
        self.analogy_entids = [int(ent) for ent in read_txt('data/analogy/analogy_ent_id')]
        self.analogy_relids = [int(rel) for rel in read_txt('data/analogy/analogy_rel_id')]

        self.data = {}
        for f in ['train', 'train_ft', 'test_ft', 'valid_ft']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

    def get_examples(self, split):
        return self.data[split+'_ft']

    def get_train(self):
        return self.data['train_ft']

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, at: Tuple[int] = (1, 3, 5, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        
        q = examples.clone()
        if n_queries > 0:
            permutation = torch.randperm(len(examples))[:n_queries]
            q = examples[permutation]
        
        ranks = []
        outputs = []
        b_begin = 0
        while b_begin < examples.shape[0]:
            input_batch = q[
                b_begin:b_begin + 500
            ].cuda()
            predictions, factors = model.forward(input_batch)
            
            _, output = torch.sort(predictions, dim=1, descending=True)
            _, predictions = torch.sort(output, dim=1)
            truth = input_batch[:, 3]
            rank = predictions[torch.arange(input_batch.shape[0]), truth].detach().cpu() + 1
            ranks.append(rank)
            outputs.append(np.array(output.detach().cpu()))
            
            b_begin += 500

        ranks = torch.cat(ranks).float()
        outputs = np.concatenate(outputs)
        mean_ranks = torch.mean(ranks).item()
        mean_reciprocal_ranks = torch.mean(1. / ranks).item()
        hits_ats = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            at
        ))))
        return mean_reciprocal_ranks, mean_ranks, hits_ats

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
  