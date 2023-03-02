from html import entities
import os
import torch
import random
random.seed(1)
import transformers
from PIL import Image
import pickle
from enum import Enum
from os import listdir
from dataclasses import dataclass
from typing import Any, Optional, Union
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer
from transformers.models.clip import CLIPProcessor
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)
from .base_data_module import BaseDataModule
from .processor import KGProcessor, get_dataset

transformers.logging.set_verbosity_error()


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        entity_img_files: {entity: entity_img_files}
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    num_labels: int = 0

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        features_keys = {}
        label = [feature.pop("label") for feature in features]
        rel_label = [feature.pop("rel_label") for feature in features] if "rel_label" in features[0].keys() else None
        head_ent = [feature.pop("head_ent") for feature in features] if "head_ent" in features[0].keys() else None
        tail_ent = [feature.pop("tail_ent") for feature in features] if "tail_ent" in features[0].keys() else None
        pre_type = [feature.pop("pre_type") for feature in features] if "pre_type" in features[0].keys() else None
        rel_idx = [feature.pop("rel_idx") for feature in features] if "rel_idx" in features[0].keys() else None
        sep_idx = [feature.pop("sep_idx") for feature in features] if "sep_idx" in features[0].keys() else None
        q_head_idx = [feature.pop("q_head_idx") for feature in features] if "q_head_idx" in features[0].keys() else None
        a_head_idx = [feature.pop("a_head_idx") for feature in features] if "a_head_idx" in features[0].keys() else None

        for k in features[0].keys():
            # ignore the padding arguments
            if k in ["input_ids", "attention_mask", "token_type_ids"]: continue
            features_keys[k] = [feature.pop(k) for feature in features]

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        pixel_images, visual_attention_masks = [], []

        for head, tail in zip(head_ent, tail_ent):
            pixel_image, visual_attention_mask = [], []
            if head and tail:   # select one head and one tail
                if isinstance(entity2visual_features, torch.Tensor):
                    pixel_image.append(entity2visual_features[entities.index(head)])
                    pixel_image.append(entity2visual_features[entities.index(tail)])
                elif isinstance(entity2visual_features, dict):
                    pixel_image.append(torch.from_numpy(entity2visual_features[head]).squeeze())
                    pixel_image.append(torch.from_numpy(entity2visual_features[tail]).squeeze())
                    visual_attention_mask.append(torch.ones((36), dtype=torch.float))
                    visual_attention_mask.append(torch.ones((36), dtype=torch.float))
            else:               # select two head or two tail
                entity = head if head is not None else tail
                if isinstance(entity2visual_features, torch.Tensor):
                    # MKG former
                    if entity:
                        pixel_image.append(entity2visual_features[entities.index(entity)])
                    else:
                        pixel_image.append(torch.zeros(entity2visual_features.shape[1:]))
                    pixel_image.append(torch.zeros(entity2visual_features.shape[1:]))
                    
                elif isinstance(entity2visual_features, dict):
                    # visualbert & vilbert
                    if entity:
                        pixel_image.append(torch.from_numpy(entity2visual_features[entity]).squeeze())
                        visual_attention_mask.append(torch.ones((36), dtype=torch.float))
                    else:
                        pixel_image.append(torch.zeros((36, 2048)))
                        visual_attention_mask.append(torch.zeros((36), dtype=torch.float))
                    pixel_image.append(torch.zeros((36, 2048)))
                    visual_attention_mask.append(torch.zeros((36), dtype=torch.float))
                    
            if isinstance(entity2visual_features, torch.Tensor):
                pixel_images.append(torch.stack(pixel_image))
            elif isinstance(entity2visual_features, dict):
                pixel_images.append(torch.cat(pixel_image)) # 72, 2048
                visual_attention_masks.append(torch.cat(visual_attention_mask))
                
        features['pixel_values'] = torch.stack(pixel_images)    # (bsz, 2, 3, 224, 224) or (bsz, 72, 2048)
        features['label'] = torch.tensor(label)
        if pre_type:
            features['pre_type'] = torch.tensor(pre_type)
        if rel_idx:
            features['rel_idx'] = torch.tensor(rel_idx)
        if sep_idx:
            features['sep_idx'] = torch.tensor(sep_idx)
            # features['attention_mask'] = features['attention_mask'].unsqueeze(1).expand([features['input_ids'].size(0), features['input_ids'].size(1), features['input_ids'].size(1)]).clone()
            # for i, idx in enumerate(sep_idx):
            #     features['attention_mask'][i, :idx[2], idx[2]:] = 0
        if len(visual_attention_masks) > 0:
            features['visual_attention_mask'] = torch.stack(visual_attention_masks)
        if rel_label:
            features['rel_label'] = torch.tensor(rel_label)
        if q_head_idx:
            features['q_head_idx'] = torch.tensor(q_head_idx)
        if a_head_idx:
            features['a_head_idx'] = torch.tensor(a_head_idx)

        features.update(features_keys)
        return features


class KGC(BaseDataModule):
    def __init__(self, args, model) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)
        self.processor = KGProcessor(self.tokenizer, args)
        self.label_list = self.processor.get_labels(args.data_dir)
        entity_list = self.processor.get_entities(args.data_dir)
        print(len(entity_list))
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_list})
        
        global entities
        with open(self.processor.entity_path, 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip().split("\t")[0])
        global entity2visual_features
        if args.model_class == 'VisualBertKGC' or  args.model_class == 'VilBertKGC':
            file = open(os.path.join(args.data_dir, 'analogy_entity2vec.pickle'), 'rb')
            entity2visual_features = pickle.load(file)
            file.close()
        elif args.model_class == 'ViltKGC':
            entity2visual_features = torch.load(os.path.join(args.data_dir, 'entity_image_features_vilt.pth'))            # 3, 384, 384
        else:
            entity2visual_features = torch.load(os.path.join(args.data_dir, 'entity_image_features.CLIP-VIT-16-32.pth'))  # 3, 224, 224
            
        self.sampler = DataCollatorForSeq2Seq(
                        self.tokenizer,
                        model=model,
                        label_pad_token_id=self.tokenizer.pad_token_id,
                        pad_to_multiple_of=8 if self.args.precision == 16 else None,
                        padding="longest",
                        max_length=self.args.max_seq_length,
                        num_labels=len(entity_list),
                    )
        relations_tokens = self.processor.get_relations(args.data_dir)
        self.num_relations = len(relations_tokens)
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relations_tokens})

        vocab = self.tokenizer.get_added_vocab()    # dict: word: idx
        self.relation_id_st = vocab[relations_tokens[0]]
        self.relation_id_ed = vocab[relations_tokens[-1]] + 1
        self.entity_id_st = vocab[entity_list[0]]
        self.entity_id_ed = vocab[entity_list[-1]] + 1
        
        # analogy entities and relations
        analogy_entities = self.processor.get_analogy_entities(args.data_dir)
        analogy_relations = self.processor.get_analogy_relations(args.data_dir)
        self.analogy_entity_ids = [vocab[ent] for ent in analogy_entities]
        self.analogy_relation_ids = [vocab[rel] for rel in analogy_relations]


    def setup(self, stage=None):
        self.data_train = get_dataset(self.args, self.processor, "train")
        self.data_val = get_dataset(self.args, self.processor, "dev")
        self.data_test = get_dataset(self.args, self.processor, "test")

    def prepare_data(self):
        pass

    def get_config(self):
        d = {}
        for k, v in self.__dict__.items():
            if "st" in k or "ed" in k or 'analogy' in k:
                d.update({k:v})
        
        return d

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--data_dir", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--warm_up_radio", type=float, default=0.1, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        return parser

    def get_tokenizer(self):
        return self.tokenizer

    def train_dataloader(self):
        return DataLoader(self.data_train, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)

