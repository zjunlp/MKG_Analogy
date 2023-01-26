import os
import sys
import csv
import json
import torch
import pickle
import logging
import inspect
import random
random.seed(1)
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

logger = logging.getLogger(__name__)


def lmap(a, b):
    return list(map(a,b))


def cache_results(_cache_fp, _refresh=False, _verbose=1):
    def wrapper_(func):
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', '_refresh', '_verbose'):
                raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

        def wrapper(*args, **kwargs):
            my_args = args[0]
            mode = args[-1]
            if '_cache_fp' in kwargs:
                cache_filepath = kwargs.pop('_cache_fp')
                assert isinstance(cache_filepath, str), "_cache_fp can only be str."
            else:
                cache_filepath = _cache_fp
            if '_refresh' in kwargs:
                refresh = kwargs.pop('_refresh')
                assert isinstance(refresh, bool), "_refresh can only be bool."
            else:
                refresh = _refresh
            if '_verbose' in kwargs:
                verbose = kwargs.pop('_verbose')
                assert isinstance(verbose, int), "_verbose can only be integer."
            else:
                verbose = _verbose
            refresh_flag = True
            
            model_name = my_args.model_name_or_path.split("/")[-1]
            is_pretrain = my_args.pretrain
            cache_filepath = os.path.join(my_args.data_dir, f"cached_{mode}_features{model_name}_pretrain{is_pretrain}.pkl")
            refresh = my_args.overwrite_cache

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = pickle.load(f)
                    if verbose == 1:
                        logger.info("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    with open(cache_filepath, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_


def solve(line,  set_type="train", pretrain=1):
    """_summary_

    Args:
        line (_type_): (head, relation, tail), wiki q_id, if pretrain=1: head = tail, relation = rel[0]
        set_type (str, optional): _description_. Defaults to "train".
        pretrain (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    examples = []

    guid = "%s-%s" % (set_type, 0)
    
    if pretrain:
        head, rel, tail = line
        relation_text = rel2text[rel]
        
        rnd = random.random()
        if rnd <= 0.4:
            # (T, T) -> (head, rel, MASK)
            head_ent_text = ent2text[head]
            tail_ent_text = ent2text[tail]
            head_ent = None
            tail_ent = None
            
        elif rnd > 0.4 and rnd < 0.7:
            # (I, T)
            head_ent_text = ""
            tail_ent_text = ent2text[tail]
            head_ent = head
            tail_ent = None
        else:
            # (I, I)
            head_ent_text = ""
            tail_ent_text = ""
            head_ent = head
            tail_ent = tail
        
        # (head, rel, MASK)
        examples.append(
                InputExample(
                    guid=guid,
                    text_a="[UNK] " + head_ent_text, 
                    text_b="[PAD] " + relation_text,
                    text_c="[MASK]", 
                    real_label=ent2id[tail],
                    head_id=ent2id[head], 
                    head_ent=head_ent, 
                    pre_type=1,
                )
            )
        # (head, MASK, tail)
        examples.append(
            InputExample(
                guid=guid,
                text_a="[UNK] " + head_ent_text, 
                text_b="[MASK]",
                text_c="[UNK] " + tail_ent_text, 
                real_label=rel2id[rel],
                head_id=ent2id[head], 
                head_ent=head_ent, 
                tail_ent=tail_ent,
                pre_type=2,
            )
        )
    else:
        head, rel, tail = line['example'][0], line['relation'], line['example'][1]
        question, answer = line['question'], line['answer']
        mode = line['mode']
        
        if mode == 0:
            head_ent_text, tail_ent_text = ent2text[head], ent2text[tail]
            # (T1, T2) -> (I1, ?)
            examples.append(
                AnalogyInputExample(
                    guid=guid, 
                    text_a="[UNK] " + head_ent_text, 
                    text_b="[PAD]",
                    text_c="[UNK] " + tail_ent_text, 
                    text_d="[UNK] ",
                    text_e="[PAD]",
                    text_f="[MASK]",
                    real_label=analogy_ent2id[answer], 
                    relation=rel2id[rel],
                    q_head_id=ent2id[head], 
                    q_tail_id=ent2id[tail], 
                    a_head_id=ent2id[question],
                    head_ent=question, 
                    tail_ent=None
                )
            )
        elif mode == 1:
            head_ent_text = ent2text[question]
            # (I1, I2) -> (T1, ?)
            examples.append(
                AnalogyInputExample(
                    guid=guid, 
                    text_a="[UNK] ", 
                    text_b="[PAD]", 
                    text_c="[UNK] ", 
                    text_d="[UNK] " + head_ent_text,
                    text_e="[PAD]", 
                    text_f="[MASK]",
                    real_label=analogy_ent2id[answer], 
                    relation=rel2id[rel],
                    q_head_id=ent2id[head], 
                    q_tail_id=ent2id[tail], 
                    a_head_id=ent2id[question],
                    head_ent=head, 
                    tail_ent=tail,
                )
            )
        elif mode == 2:
            tail_ent_text = ent2text[tail]
            # (I1, T1) -> (I2, ?)
            examples.append(
                AnalogyInputExample(
                    guid=guid, 
                    text_a="[UNK] ", 
                    text_b="[PAD]", 
                    text_c="[UNK] " + tail_ent_text, 
                    text_d="[UNK] ",
                    text_e="[PAD]", 
                    text_f="[MASK]",
                    real_label=analogy_ent2id[answer], 
                    relation=rel2id[rel],
                    q_head_id=ent2id[head], 
                    q_tail_id=ent2id[tail], 
                    a_head_id=ent2id[question],
                    head_ent=head, 
                    tail_ent=question,
                )
            )
    return examples


def filter_init(t1,t2, ent2id_, ent2token_, rel2id_, analogy_ent2id_, analogy_rel2id_):
    global ent2text
    global rel2text
    global ent2id
    global ent2token
    global rel2id
    global analogy_ent2id
    global analogy_rel2id

    ent2text =t1
    rel2text =t2
    ent2id = ent2id_
    ent2token = ent2token_
    rel2id = rel2id_
    analogy_ent2id = analogy_ent2id_
    analogy_rel2id = analogy_rel2id_

def delete_init(ent2text_):
    global ent2text
    ent2text = ent2text_


@cache_results(_cache_fp="./dataset")
def get_dataset(args, processor, mode):

    assert mode in ["train", "dev", "test"], "mode must be in train dev test!"

    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    
    features = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    encoder = MultiprocessingEncoder(tokenizer, args)
    pool = Pool(16, initializer=encoder.initializer)
    encoder.initializer()
    encoded_lines = pool.imap(encoder.encode_lines, examples, 1000)

    for enc_lines in tqdm(encoded_lines, total=len(examples)):
        for enc_line in enc_lines:
            features.append(enc_line)

    num_entities = len(processor.get_entities(args.data_dir))
    num_relations = len(processor.get_relations(args.data_dir))
    if args.pretrain:
        for f_id, feature in enumerate(features):
            head_id, rel_id, tail_id = feature.pop('head_id'), feature.pop('rel_id'), feature.pop('tail_id')
            if head_id != -1 and  tail_id != -1:
                # relation prediction
                count = 0
                entity_id = [head_id, tail_id]
                for i, ids in enumerate(feature['input_ids']):
                    if ids == tokenizer.unk_token_id and count < 2:
                        features[f_id]['input_ids'][i] = entity_id[count] + len(tokenizer)
                        count += 1
            else:
                # link prediction
                entity_id = head_id if head_id != -1 else tail_id
                for i, ids in enumerate(feature['input_ids']):
                    if ids == tokenizer.unk_token_id:
                        features[f_id]['input_ids'][i] = entity_id + len(tokenizer)
                        break
            
            for i, ids in enumerate(feature['input_ids']):
                if ids == tokenizer.pad_token_id:
                    features[f_id]['input_ids'][i] = rel_id + len(tokenizer) + num_entities
                    break
    else:
        for f_id, feature in enumerate(features):
            q_head_id, q_tail_id, a_head_id = feature.pop('q_head_id'), feature.pop('q_tail_id'), feature.pop('a_head_id')
            count = 0
            entity_id = [q_head_id, q_tail_id, a_head_id]
            rel_idx, sep_idx = [], []
            for i, ids in enumerate(feature['input_ids']):
                if count < 3 and ids == tokenizer.unk_token_id:
                    features[f_id]['input_ids'][i] = entity_id[count] + len(tokenizer)
                    if count == 0:
                        q_head_idx = i
                    elif count == 2:
                        a_head_idx = i
                    count += 1
                if ids == tokenizer.sep_token_id:
                    sep_idx.append(i)

            features[f_id]['sep_idx'] = sep_idx
            features[f_id]['q_head_idx'] = q_head_idx
            features[f_id]['a_head_idx'] = a_head_idx
            rel_id = features[f_id]['rel_label']
            
            for i, ids in enumerate(feature['input_ids']):
                if ids == tokenizer.pad_token_id:
                    features[f_id]['input_ids'][i] = len(tokenizer) + num_entities + num_relations
                    rel_idx.append(i)
                
            features[f_id]['rel_idx'] = rel_idx
    features = KGCDataset(features)
    return features


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
            self,
            guid, 
            text_a, 
            text_b=None, 
            text_c=None, 
            pre_type=0, 
            real_label=None, 
            head_id=-1, 
            rel_id=-1,
            tail_id=-1, 
            head_ent=None,
            tail_ent=None,
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            pre_type: The type of the pretrained example, 0 is (des, is, MASK) , 1 is (head, rel, MASK), 2 is (head, MASK, tail)
            text_a: string. The untokenized text of the first sequence. For single
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
           
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.pre_type = pre_type
        self.real_label = real_label
        self.head_id = head_id
        self.rel_id = rel_id    # rel id
        self.tail_id = tail_id
        self.head_ent = head_ent
        self.tail_ent = tail_ent


class AnalogyInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
            self,
            guid, 
            text_a, 
            text_b=None, 
            text_c=None, 
            text_d=None,
            text_e=None,
            text_f=None,
            real_label=None,
            relation=None,
            q_head_id=-1, 
            q_tail_id=-1, 
            a_head_id=-1,
            head_ent=None,
            tail_ent=None,
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            pre_type: The type of the pretrained example, 0 is (des, is, MASK) , 1 is (head, rel, MASK), 2 is (head, MASK, tail)
            text_a: string. The untokenized text of the first sequence. For single
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
           
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.text_d = text_d
        self.text_e = text_e
        self.text_f = text_f
        self.real_label = real_label
        self.relation = relation
        self.q_head_id = q_head_id       # entity id
        self.q_tail_id = q_tail_id
        self.a_head_id = a_head_id
        self.head_ent = head_ent     # entity des
        self.tail_ent = tail_ent


@dataclass
class InputFeatures:
    """A single set of features of data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor = None
    head_id: torch.Tensor = -1
    rel_id: torch.Tensor = -1
    tail_id: torch.Tensor = -1
    head_ent: torch.Tensor = None
    tail_ent: torch.Tensor = None
    pre_type: torch.Tensor = None
    token_type_ids: torch.Tensor = None

@dataclass
class AnalogyInputFeatures:
    """A single set of features of data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    label: torch.Tensor = None
    rel_label: torch.Tensor = None
    q_head_id: torch.Tensor = -1
    q_tail_id: torch.Tensor = -1
    a_head_id: torch.Tensor = -1
    head_ent: torch.Tensor = None
    tail_ent: torch.Tensor = None


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    
    @classmethod
    def _read_txt(cls, input_file, quotechar='\t'):
        """Reads a `quotechar` separated txt file."""
        read_lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                head, rel, tail = line.split(quotechar)
                read_lines.append((head, rel, tail.replace('\n', '')))
        return read_lines

    @classmethod
    def _read_dict_txt(cls, input_file, quotechar='\t'):
        """Reads a `quotechar` separated txt file."""
        read_dict = {}
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.split(quotechar)
                read_dict[key] = value[:-1]
        return read_dict

    @classmethod
    def _read_json(cls, input_file, quotechar='\t'):
        """Reads a `quotechar` separated txt file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            read_lines = [json.loads(line) for line in lines]
        return read_lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self, tokenizer, args):
        self.labels = set()
        self.tokenizer = tokenizer
        self.args = args
        self.entity_path = os.path.join(self.args.pretrain_path, "entity2textlong.txt") if os.path.exists(os.path.join(self.args.pretrain_path, 'entity2textlong.txt')) \
        else os.path.join(self.args.pretrain_path, "entity2text.txt")

    def get_train_examples(self, data_dir):
        """See base class."""
        if self.args.pretrain:
            return self._create_examples(
                self._read_txt(os.path.join(self.args.pretrain_path, "wiki_tuple_ids.txt")), "train", data_dir, self.args)
        else:
            return self._create_examples(
                self._read_json(os.path.join(data_dir, "train.json")), "train", data_dir, self.args)


    def get_dev_examples(self, data_dir):
        """See base class."""
        if self.args.pretrain:
            return self._create_examples(
                self._read_txt(os.path.join(self.args.pretrain_path, "wiki_tuple_ids.txt")), "dev", data_dir, self.args)
        else:
            return self._create_examples(
                self._read_json(os.path.join(data_dir, "dev.json")), "dev", data_dir, self.args)

    def get_test_examples(self, data_dir, chunk=""):
        """See base class."""
        if self.args.pretrain:
            return self._create_examples(
                self._read_txt(os.path.join(self.args.pretrain_path, "wiki_tuple_ids.txt")), "test", data_dir, self.args)
        else:
            return self._create_examples(
                self._read_json(os.path.join(data_dir, "test.json")), "test", data_dir, self.args)


    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(self.args.pretrain_path, "relation2text.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip().split('\t')[0])
        rel2token = {ent : f"[RELATION_{i}]" for i, ent in enumerate(relations)}
        return list(rel2token.values())
    
    def get_analogy_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(self.args.pretrain_path, "relation2text.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip().split('\t')[0])
                
        with open(os.path.join(data_dir, "analogy_relations.txt"), 'r') as f:
            lines = f.readlines()
            analogy_relations = []
            for line in lines:
                analogy_relations.append(line.strip().replace('\n', ''))
                
        rel2token = {ent : f"[RELATION_{i}]" for i, ent in enumerate(relations) if ent in analogy_relations}
        return list(rel2token.values())

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        with open(self.entity_path, 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip().split("\t")[0])
        
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        return list(ent2token.values())
    
    def get_analogy_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        with open(self.entity_path, 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip().split("\t")[0])
                
        with open(os.path.join(data_dir, "analogy_entities.txt"), 'r') as f:
            lines = f.readlines()
            analogy_entities = []
            for line in lines:
                analogy_entities.append(line.strip().replace('\n', ''))
        
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities) if ent in analogy_entities}
        return list(ent2token.values())
    
    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        relation = []
        with open(os.path.join(self.args.pretrain_path, "relation2text.txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                relation.append(line.strip().split("\t")[-1])
        return relation

    def _create_examples(self, lines, set_type, data_dir, args):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = self._read_dict_txt(self.entity_path, quotechar='\t')
        # entities
        entities = list(ent2text.keys())
        # entity to virtual token
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        # entity to id
        ent2id = {ent : i for i, ent in enumerate(entities)}
        
        # relation to text
        rel2text = self._read_dict_txt(os.path.join(self.args.pretrain_path, "relation2text.txt"))
        
        # rel id -> relation token id
        rel2id = {rel: i for i, rel in enumerate(rel2text.keys())}

        # anlogy entities and relations
        with open(os.path.join(data_dir, "analogy_entities.txt"), 'r') as f:
            analogy_entities = []
            for line in f.readlines():
                analogy_entities.append(line.strip().replace('\n', ''))
        analogy_ent2id, i = {}, 0
        for ent in entities:
            if ent in analogy_entities:
                analogy_ent2id[ent] = i
                i += 1
        
        with open(os.path.join(data_dir, "analogy_relations.txt"), 'r') as f:
            analogy_relations = []
            for line in f.readlines():
                analogy_relations.append(line.strip().replace('\n', ''))
        analogy_rel2id, i = {}, 0
        for rel in rel2id:
            if rel in analogy_relations:
                rel2id[rel] = i
                i += 1

        examples = []
        filter_init(ent2text, rel2text, ent2id, ent2token, rel2id, analogy_ent2id, analogy_rel2id)

        if args.pretrain:
            # delete entities without text name.
            tmp_lines = []
            not_in_text = 0
            for line in tqdm(lines, desc="delete entities without text name."):
                if (line[0] not in ent2text) or (line[2] not in ent2text) or (line[1] not in rel2text):
                    not_in_text += 1
                    continue
                tmp_lines.append(line)
            lines = tmp_lines
            print(f"total entity not in text : {not_in_text} ")

            examples = list(
                tqdm(
                    map(partial(solve, pretrain=self.args.pretrain), lines),
                    total=len(lines),
                    desc="Pretrain pre_type=1, convert text to examples"
                )
            )

        else:
            examples = list(
                tqdm(
                    map(partial(solve, pretrain=self.args.pretrain), lines),
                    total=len(lines),
                    desc="Fine-tuning, convert text to examples"
                )
            )

        # flatten examples
        examples = [sub_example for example in examples for sub_example in example]
        # delete vars
        del ent2text, rel2text, ent2id, ent2token, rel2id
        return examples


class KGCDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        return self.features[index]
    
    def __len__(self):
        return len(self.features)


class MultiprocessingEncoder(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.pretrain = args.pretrain
        self.max_seq_length = args.max_seq_length

    def initializer(self):
        global bpe
        bpe = self.tokenizer

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        lines: [InputExamples]
        """
        enc_lines = []
        enc_lines.append(self.convert_examples_to_features(example=lines))
        return enc_lines

    def convert_examples_to_features(self, example):
        pretrain = self.pretrain
        max_seq_length = self.max_seq_length
        global bpe
        """Loads a data file into a list of `InputBatch`s."""
        
        text_a = example.text_a
        text_b = example.text_b
        text_c = example.text_c
        
        if self.pretrain:
            input_text_a = bpe.sep_token.join([text_a, text_b, text_c])
            input_text_b = None
            inputs = bpe(
                input_text_a,
                input_text_b,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )
            features = asdict(InputFeatures(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs['attention_mask'],
                            token_type_ids=inputs['token_type_ids'],
                            label=example.real_label,
                            head_id=example.head_id,
                            rel_id=example.rel_id,
                            tail_id=example.tail_id,
                            head_ent=example.head_ent,
                            tail_ent=example.tail_ent,
                            pre_type=example.pre_type,
                        )
                    )
        else:
            text_d, text_e, text_f = example.text_d, example.text_e, example.text_f
            
            input_text_a = bpe.sep_token.join([text_a, text_b, text_c])
            input_text_b = bpe.sep_token.join([text_d, text_e, text_f])
        
            inputs = bpe(
                input_text_a,
                input_text_b,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )
            
            features = asdict(AnalogyInputFeatures(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs['attention_mask'],
                            token_type_ids=inputs['token_type_ids'],
                            label=example.real_label,
                            rel_label=example.relation,
                            q_head_id=example.q_head_id,
                            q_tail_id=example.q_tail_id,
                            a_head_id=example.a_head_id,
                            head_ent=example.head_ent,
                            tail_ent=example.tail_ent
                        )
                    )
        assert bpe.mask_token_id in inputs.input_ids, "mask token must in input"

        return features
