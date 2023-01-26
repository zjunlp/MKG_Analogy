import argparse
from typing import Dict
import os
import torch
from torch import optim

from datasets import Dataset, AnalogyDataset
from models import CP, ComplEx, Analogy, read_txt
from regularizers import F2, N3
from optimizers import KBCOptimizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


big_datasets = ['FB15K', 'analogy']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx', 'Analogy']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
parser.add_argument(
    '--finetune', default=False, action="store_true",
    help="whether finetune"
)
parser.add_argument(
    '--ckpt', default=None, type=str,
    help="checkpoint path"
)
args = parser.parse_args()
print(args)

if args.finetune:
    dataset = AnalogyDataset(args.dataset)
else:
    dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

print(dataset.get_shape())
model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init, args.finetune),
    'Analogy': lambda: Analogy(dataset.get_shape(), args.rank, args.init, args.finetune),
}[args.model]()

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

if args.finetune:
    if args.ckpt is not None:
        if args.model == 'ComplEx':
            model.load_state_dict(torch.load(args.ckpt))
        else:
            model.load_state_dict(torch.load(args.ckpt))
    model.finetune = True
    model.analogy_entids = [int(ent) for ent in read_txt('data/analogy/analogy_ent_id')]
    model.analogy_relids = [int(rel) for rel in read_txt('data/analogy/analogy_rel_id')]
    print('Successful load model state dict!')
    
device = 'cuda'
model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size, finetune=args.finetune)


def avg_both(mrrs: Dict[str, float],mrs, hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'MR': mrs, 'hits@[1,3,5,10]': h}

def avg_both_ft(mrrs: Dict[str, float], mrs, hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    return {'MRR': mrrs, 'MR':mrs, 'hits@[1,3,5,10]': hits}


if not args.finetune:
    cur_loss = 0
    best_test_mrr = 1e-8
    curve = {'train': [], 'valid': [], 'test': []}
    for e in range(args.max_epochs):
        cur_loss = optimizer.epoch(examples)

        if (e + 1) % args.valid == 0:
            valid, test, train = [
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test', 'train']
            ]

            curve['valid'].append(valid)
            curve['test'].append(test)
            curve['train'].append(train)
            
            if test['MRR'] > best_test_mrr:
                best_test_mrr = test['MRR']
                torch.save(model.state_dict(), 'checkpoint/pt_best_model_analogy.pth')

            print("\t TRAIN: ", train)
            print("\t TEST : ", test)
            print("\t VALID : ", valid)

    results = dataset.eval(model, 'test', -1)
    print("\n\nTEST : ", results)
else:
    cur_loss = 0
    best_test_mrr = 1e-8
    curve = {'train_ft': [], 'valid_ft': [], 'test_ft': []}
    for e in range(args.max_epochs):
        cur_loss = optimizer.epoch(examples)

        if (e + 1) % args.valid == 0:
            valid, test, train = [
                avg_both_ft(*dataset.eval(model, split, -1 if split != 'train' else 50000)) for split in ['valid', 'test', 'train']
            ]

            curve['valid_ft'].append(valid)
            curve['test_ft'].append(test)
            curve['train_ft'].append(train)
            
            if test['MRR'] > best_test_mrr:
                best_test_mrr = test['MRR']
                torch.save(model.state_dict(), f'checkpoint/ft_best_model_{args.model}_{args.learning_rate}.pth')

            print("\t TRAIN_FT: ", train)
            print("\t TEST_FT: ", test)
            print("\t VALID_FT: ", valid)

    results = dataset.eval(model, 'test', -1)
    print("\n\nTEST_FT: ", results)
    
