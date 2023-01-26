import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup
from functools import partial
from .utils import LabelSmoothSoftmaxCEV1
from typing import Callable, Iterable, List

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))

class TransformerLitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None, data_config={}):
        super().__init__(model, args)
        self.save_hyperparameters(args)
        if args.label_smoothing != 0.0:
            self.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=args.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.best_acc = 0
        self.first = True
        self.tokenizer = tokenizer
        self.__dict__.update(data_config)   # update config

        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.alpha = args.alpha

        # if args.pretrain:
        #     # when pretrain, only tune embedding layers
        #     self._freeze_attention()
        # self._freeze_word_embedding()
        
    def _init_relation_word(self):
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': ["[R]"]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decode = partial(decode, tokenizer=self.tokenizer)
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            continous_label_word = self.analogy_relation_ids
            
            rel_word = [a[0] for a in self.tokenizer(["[R]"], add_special_tokens=False)['input_ids']]
            for i, idx in enumerate(rel_word):
                word_embeddings.weight[rel_word[i]] = torch.mean(word_embeddings.weight[continous_label_word], dim=0)
            
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)

    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self) -> None:
        self._freeze_word_embedding()
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        label = batch.pop("label")
        rel_label = batch.pop('rel_label') if 'rel_label' in batch else None
        pre_type = batch.pop('pre_type') if 'pre_type' in batch else None
        rel_idx = batch.pop('rel_idx') if 'rel_idx' in batch else None
        q_head_idx = batch.pop('q_head_idx') if 'q_head_idx' in batch else None
        a_head_idx = batch.pop('a_head_idx') if 'a_head_idx' in batch else None
        
        input_ids = batch['input_ids']
        model_output = self.model(**batch, return_dict=True)
        logits = model_output[0].logits
        bs = input_ids.shape[0]

        if self.args.pretrain:
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            assert mask_idx.shape[0] == bs, "only one mask in sequence!"
            mask_logits = logits[torch.arange(bs), mask_idx]    # bsz, 1, vocab
        
            entity_loss, relation_loss = 0, 0
            entity_mask = (pre_type != 2).nonzero(as_tuple=True)[0]
            if len(entity_mask) > 0:
                entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]
                entity_label = label[entity_mask]
                entity_loss = self.loss_fn(entity_logits, entity_label)

            relation_mask = (pre_type == 2).nonzero(as_tuple=True)[0]
            if len(relation_mask) > 0:
                relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
                relation_label = label[relation_mask]
                relation_loss = self.loss_fn(relation_logits, relation_label)
            
            loss = entity_loss + relation_loss
            
        else:
            # tail prediction
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)    # bsz
            mask_logits = logits[torch.arange(bs), mask_idx][:, self.analogy_entity_ids]    # bsz, 1, entity
            
            """relaxation loss: close relation and far entity"""
            if 'VisualBertKGC' in str(type(self.model)):
                img_len = batch['pixel_values'].shape[1]
                rel_idx = rel_idx + img_len
                q_head_idx = q_head_idx + img_len
                a_head_idx = a_head_idx + img_len
            trans_hidden_states = model_output[1]   # bsz, len, hidden_size
            rel_hidden_state = trans_hidden_states[torch.arange(bs), rel_idx[torch.arange(bs), 0]] # relation between examples
            r_hidden_state = trans_hidden_states[torch.arange(bs), rel_idx[torch.arange(bs), 1]]   # relation between question and answer
            q_head_hidden_state = trans_hidden_states[torch.arange(bs), q_head_idx[torch.arange(bs)]] # relation between examples
            a_head_hidden_state = trans_hidden_states[torch.arange(bs), a_head_idx[torch.arange(bs)]]   # relation between question and answer
            sim_loss = (F.relu(F.cosine_similarity(q_head_hidden_state, a_head_hidden_state)) + 1 - F.cosine_similarity(rel_hidden_state, r_hidden_state)).mean(0)
            loss = self.loss_fn(mask_logits, label) + self.alpha * sim_loss
            
        if batch_idx == 0:
            print('\n'.join(self.decode(batch['input_ids'][:4])))
        return loss

    def _eval(self, batch, batch_idx):
        label = batch.pop('label')  # bsz
        pre_type = batch.pop('pre_type') if 'pre_type' in batch else None
        rel_idx = batch.pop('rel_idx') if 'rel_idx' in batch else None
        rel_label = batch.pop('rel_label') if 'rel_label' in batch else None
        q_head_idx = batch.pop('q_head_idx') if 'q_head_idx' in batch else None
        a_head_idx = batch.pop('a_head_idx') if 'a_head_idx' in batch else None
        
        input_ids = batch['input_ids']

        model_output = self.model(**batch, return_dict=True)
        logits = model_output[0].logits   # bsz, len, vocab
        bsz = input_ids.shape[0]
        
        if self.args.pretrain:
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)    # bsz
            mask_logits = logits[torch.arange(bsz), mask_idx]       # bsz, vocab
        
            entity_ranks, relation_ranks = None, None
            entity_mask = (pre_type != 2).nonzero(as_tuple=True)[0]
            if len(entity_mask) > 0:
                entity_logits = mask_logits[entity_mask, self.entity_id_st:self.entity_id_ed]   # bsz, entities
                entity_label = label[entity_mask]
                _, entity_outputs = torch.sort(entity_logits, dim=1, descending=True)           # bsz, entities   index
                _, entity_outputs = torch.sort(entity_outputs, dim=1)
                entity_ranks = entity_outputs[torch.arange(entity_mask.size(0)), entity_label].detach().cpu() + 1
            relation_mask = (pre_type == 2).nonzero(as_tuple=True)[0]
            if len(relation_mask) > 0:
                relation_logits = mask_logits[relation_mask, self.relation_id_st:self.relation_id_ed]
                relation_label = label[relation_mask]
                _, relation_outputs = torch.sort(relation_logits, dim=1, descending=True) # bsz, relations   index
                _, relation_outputs = torch.sort(relation_outputs, dim=1)
                relation_ranks = relation_outputs[torch.arange(relation_mask.size(0)), relation_label].detach().cpu() + 1
            
            if entity_ranks is not None and relation_ranks is not None:
                return dict(entity_ranks=np.array(entity_ranks), relation_ranks=np.array(relation_ranks))
            elif entity_ranks is not None:
                return dict(entity_ranks=np.array(entity_ranks))
            elif relation_ranks is not None:
                return dict(relation_ranks=np.array(relation_ranks))
            else:
                raise ValueError('entity and relation cannot be None at the same time.')
            
        else:
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)    # bsz
            mask_logits = logits[torch.arange(bsz), mask_idx][:, self.analogy_entity_ids]    # bsz, 1, entity
            
            _, outputs1 = torch.sort(mask_logits, dim=1, descending=True)         
            _, outputs = torch.sort(outputs1, dim=1)                              
            entity_ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
            
            return dict(entity_ranks=np.array(entity_ranks))


    def validation_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs) -> None:
        entity_ranks = [_['entity_ranks'] for _ in outputs if 'entity_ranks' in _]
        if len(entity_ranks) > 0:
            entity_ranks = np.concatenate(entity_ranks)

            # entity
            hits20 = (entity_ranks<=20).mean()
            hits10 = (entity_ranks<=10).mean()
            hits5 = (entity_ranks<=5).mean()
            hits3 = (entity_ranks<=3).mean()
            hits1 = (entity_ranks<=1).mean()

            self.log("Eval_entity/hits1", hits1)
            self.log("Eval_entity/hits3", hits3)
            self.log("Eval_entity/hits5", hits5)
            self.log("Eval_entity/hits10", hits10)
            self.log("Eval_entity/hits20", hits20)
            self.log("Eval_entity/mean_rank", entity_ranks.mean())
            self.log("Eval_entity/mrr", (1. / entity_ranks).mean())
            self.log("entity_hits10", hits10, prog_bar=True)
            self.log("entity_hits1", hits1, prog_bar=True)
            
    
    def test_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        # self.log("Test/ranks", np.mean(ranks))
        return result

    def test_epoch_end(self, outputs) -> None:
        entity_ranks = [_['entity_ranks'] for _ in outputs if 'entity_ranks' in _]

        if len(entity_ranks) > 0:
            entity_ranks = np.concatenate(entity_ranks)

            # entity
            hits20 = (entity_ranks<=20).mean()
            hits10 = (entity_ranks<=10).mean()
            hits5 = (entity_ranks<=5).mean()
            hits3 = (entity_ranks<=3).mean()
            hits1 = (entity_ranks<=1).mean()

            self.log("Eval_entity/hits1", hits1)
            self.log("Eval_entity/hits3", hits3)
            self.log("Eval_entity/hits5", hits5)
            self.log("Eval_entity/hits10", hits10)
            self.log("Eval_entity/hits20", hits20)
            self.log("Eval_entity/mean_rank", entity_ranks.mean())
            self.log("Eval_entity/mrr", (1. / entity_ranks).mean())
            self.log("entity_hits10", hits10, prog_bar=True)
            self.log("entity_hits1", hits1, prog_bar=True)

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * self.args.warm_up_radio, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
    
    def _freeze_attention(self):
        for k, v in self.model.named_parameters():
            if "word" not in k:
                v.requires_grad = False
            else:
                print(k)
    
    def _freeze_word_embedding(self):
        for k, v in self.model.named_parameters():
            if "word" in k:
                print(k)
                v.requires_grad = False

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)

        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--bce", type=int, default=0, help="")
        return parser
