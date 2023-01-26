import os
import json
import errno
from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict

DATA_PATH = 'RSME'

def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['wiki_tuple_ids']
    entities, relations = set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    print("{} entities and {} relations".format(len(entities), len(relations)))
    n_relations = len(relations)
    n_entities = len(entities)
    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'wiki_tuple_ids.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()

def prepare_finetune_dataset():
    # encode finetune data
    with open('data/analogy/ent_id', 'r') as f:
        ent2id = {line.split('\t')[0]: line.split('\t')[1][:-1] for line in f.readlines()}
        
    with open('data/analogy/rel_id', 'r') as f:
        rel2id = {line.split('\t')[0]: line.split('\t')[1][:-1] for line in f.readlines()}
    
    
    for name in ['train', 'dev', 'test']:
        with open(f'../MarT/dataset/MARS/{name}_ids3.json', 'r') as f:
            datas = []
            load_data = [json.loads(line) for line in f.readlines()]
            for line in load_data:
                example_h, example_t, question, answer = line['example'][0], line['example'][1], line['question'], line['answer']
                rel, mode = line['relation'], line['mode']
                if example_h in ent2id and example_t in ent2id and question in ent2id and answer in ent2id and rel in rel2id:
                    datas.append([ent2id[example_h], ent2id[example_t], ent2id[question], ent2id[answer], rel2id[rel], mode])

        print(name, len(load_data), len(datas))
        out = open(f"data/analogy/{name}_ft.pickle", 'wb')
        pickle.dump(np.array(datas).astype('uint64'), out)
        out.close()
    
if __name__ == "__main__":
    datasets = ['Analogy']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_finetune_dataset()
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise