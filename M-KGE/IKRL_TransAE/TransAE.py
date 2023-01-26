import torch
import os
import re
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils import data
from collections import Counter
import torch.optim as optim
from tqdm import tqdm
import ctypes
import pickle
import gensim
import random
from DATA import TrainDataLoader, TestDataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def pvdm(entity2id, retrain=True, vector_size=100, min_count=2, epochs=40):
    if not retrain:
        with open("embedding_weights/textembed_" + str(len(entity2id)) + "_" + str(vector_size) + "_" + str(min_count) + "_" + str(epochs) +".pkl", "rb") as emf:
            inferred_vector_list = pickle.load(emf)
        return inferred_vector_list
    
    entity2glossary = dict()
    with open(f"{root}/entity2textlong.txt", "r") as glossf: # TODO: add datapath
        for line in glossf:
            entity, glossary = line.split("\t")
            entity2glossary[entity] = glossary
    
    entity2description = list()
    # Was Doing training on whole dataset, should not do it, should be done only on training dataset
    for entity, index in entity2id.items():
        entity2description.append(entity2glossary[entity])

    def read_corpus(tokens_only=False):
        for i, v in enumerate(entity2description):
            tokens = gensim.utils.simple_preprocess(v)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
            
    train_corpus = list(read_corpus())
    
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    inferred_vector_list = list()

    for doc_id in range(len(train_corpus)): # train_corpus is already sorted in entity2id order, so will be the saved vectors
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        # print(inferred_vector)     # inferred_vector is of size embedding_dim
        inferred_vector_list.append(inferred_vector)
        
    with open("embedding_weights/textembed_" + str(len(entity2id)) + "_" + str(vector_size) + "_" + str(min_count) + "_" + str(epochs) +".pkl", "wb+") as emf:
        pickle.dump(inferred_vector_list, emf)
    
    return inferred_vector_list


class Trainer(object):

    def __init__(self, 
                 model = None,
                 data_loader = None,
                 train_times = 1000,
                 alpha = 0.5,
                 use_gpu = True,
                 opt_method = "sgd",
                 save_steps = None,
                 checkpoint_dir = None,
                 finetune=False):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        self.finetune = finetune

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode'], 
            'task_mode': self.get_task_mode(data['batch_h'])
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_one_step_ft(self, data):
        self.optimizer.zero_grad()
        loss, ent_score = self.model({
            'e_head': data[0].cuda(),
            'e_tail': data[1].cuda(),
            'q_head': data[2].cuda(),
            'q_tail': data[3].cuda(),
            'relation': data[4].cuda(),
            'task_mode': data[5].cuda(),
            'mode': 'normal'
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_task_mode(self, batch_h):
        random_data = torch.randint(0, 10, (batch_h.shape[0],)).tolist()
        task_mode = []
        for rnd in random_data:
            if rnd < 4:
                task_mode.append(0)
            elif rnd >= 4 and rnd < 7:
                task_mode.append(1)
            else:
                task_mode.append(2)
        return torch.tensor(task_mode).cuda()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr = self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")
        
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader:
                if not self.finetune:
                    loss = self.train_one_step(data)
                else:
                    loss = self.train_one_step_ft(data)
                res += loss
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
            
            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir = None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir


class MarginLoss(nn.Module):
    def __init__(self, adv_temperature = None, margin = 6.0):
        super(MarginLoss,self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
        else:
            return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
            
    
    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()


class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "./release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode'],
            'task_mode': self.get_task_mode(data)
        })
        
    def test_one_step_ft(self, data):
        return self.model.predict({
            'e_head': data[0].cuda(),
            'e_tail': data[1].cuda(),
            'q_head': data[2].cuda(),
            'q_tail': data[3].cuda(),
            'relation': data[4].cuda(),
            'task_mode': data[5].cuda(),
            'mode': 'normal'
        })
        
    def get_task_mode(self, data):
        bsz = max(len(data['batch_h']), len(data['batch_t']))
        random_data = torch.randint(0, 10, (bsz,)).tolist()
        task_mode = []
        for rnd in random_data:
            if rnd < 4:
                task_mode.append(0)
            elif rnd >= 4 and rnd < 7:
                task_mode.append(1)
            else:
                task_mode.append(2)
        return torch.tensor(task_mode).cuda()

    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step(data_tail)
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain)

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print (hit10)
        return mrr, mr, hit10, hit3, hit1
    
    def run_analogical_reasoning(self, type_constrain=False):
        ranks = []
        outputs = []
        for data in self.data_loader:
            predictions = self.test_one_step_ft(data)   # bsz, 11292
            truth = data[3]
            _, output = torch.sort(predictions, dim=1, descending=True)
            _, predictions = torch.sort(output, dim=1)
            rank = predictions[torch.arange(truth.shape[0]), truth].detach().cpu() + 1
            ranks.append(rank)
            outputs.append(np.array(output.detach().cpu()))
            
        ranks = torch.cat(ranks).float()
        outputs = np.concatenate(outputs)
        mean_ranks = torch.mean(ranks).item()
        mean_reciprocal_ranks = torch.mean(1. / ranks).item()
        hits_ats = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            (1, 3, 5, 10)
        ))))
        return mean_reciprocal_ranks, mean_ranks, hits_ats[-1], hits_ats[-2], hits_ats[-3], hits_ats[-4]

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod


class IMG_Encoder(nn.Module):
    def __init__(self, text_embedding_dim=100, embedding_dim = 4096, dim = 200,  hidden_text_dim=200, margin = None, epsilon = None):
        super(IMG_Encoder, self).__init__()
        with open('data/analogy/entity2id.txt') as fp:
            entity2id = fp.readlines()[1:]
            entity2id = {line.split('\t')[0]: line.split('\t')[1] for line in entity2id}

        self.entity2id = entity2id
        self.activation = nn.ReLU()
        self.entity_count = len(entity2id)
        self.dim = dim
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.criterion = nn.MSELoss(reduction='mean') 
        self.raw_embedding = nn.Embedding(self.entity_count, self.dim)

        self.text_embedding = self._init_text_emb()
        self.visual_embedding = self._init_visual_emb()

        # hidden layer 1
        self.encoder_text_linear1 = nn.Sequential(
            torch.nn.Linear(text_embedding_dim, hidden_text_dim),
            self.activation
        )
        self.encoder_visual_linear1 = nn.Sequential(
                torch.nn.Linear(embedding_dim, 1024),
                self.activation
            )
        
        # hidden layer 2
        self.encoder_combined_linear = nn.Sequential(
                torch.nn.Linear(hidden_text_dim+1024, self.dim),
                self.activation
            )

        # hidden layer 3
        self.decoder_text_linear1 = nn.Sequential(
            torch.nn.Linear(self.dim, hidden_text_dim),
            self.activation
        )
        self.decoder_visual_linear1 = nn.Sequential(
                torch.nn.Linear(self.dim, 1024),
                self.activation
            )

        # output layer
        self.decoder_text_linear2 = nn.Sequential(
            torch.nn.Linear(hidden_text_dim, text_embedding_dim),
            self.activation
        )
        self.decoder_visual_linear2 = nn.Sequential(
                torch.nn.Linear(1024, embedding_dim),
                self.activation
            )
        
    def _init_text_emb(self):
        inferred_vector_list = pvdm(self.entity2id, retrain=True) # should just train on the ones in training set
        weights = np.zeros((self.entity_count + 1, self.text_embedding_dim)) # +1 to account for padding/OOKB, initialized to 0 each one
        for index in range(len(inferred_vector_list)):
            weights[index] = inferred_vector_list[index]
        weights = torch.from_numpy(weights) 
        # print(weights.shape)
        text_emb = nn.Embedding.from_pretrained(weights, padding_idx=self.entity_count)
        return text_emb
    
    def _init_visual_emb(self):
        uniform_range = 6 / np.sqrt(self.dim)
        weights = torch.empty(self.entity_count + 1, self.embedding_dim)
        nn.init.uniform_(weights, -uniform_range, uniform_range)
        no_embed = 0
        for index, entity in enumerate(self.entity2id):
            try:
                with open("data/analogy-img/" + entity + "/avg_embedding.pkl", "rb") as visef:
                    em = pickle.load(visef)
                    weights[index] = em
            except:
                no_embed = no_embed+1
                continue
        print(no_embed)
        entities_emb = nn.Embedding.from_pretrained(weights, padding_idx=self.entity_count)
        return entities_emb

    def _init_embedding(self):
        self.ent_embeddings = nn.Embedding(self.entity_count, self.embedding_dim)
        for param in self.ent_embeddings.parameters():
            param.requires_grad = False
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        weights = torch.empty(self.entity_count, self.embedding_dim)
        no_embed = 0
        for entity, index in tqdm(self.entity2id.items()):
            try:
                with open("data/analogy-img/" + entity + "/avg_embedding.pkl", "rb") as visef:
                    em = pickle.load(visef)
                    weights[index] = em
            except:
                weights[index] = self.ent_embeddings(torch.LongTensor([index])).clone().detach()
                no_embed = no_embed+1
                continue
        print(no_embed)
        entities_emb = nn.Embedding.from_pretrained(weights)

        return entities_emb

    def forward(self, entity_id, task_mode, finetune=False, is_head=False):
        v1_t = self.text_embedding(entity_id).float()
        v1_i = self.visual_embedding(entity_id)

        v2_t = self.encoder_text_linear1(v1_t)      # bsz, 200
        v2_i = self.encoder_visual_linear1(v1_i)    # bsz, 1024
        
        '''split
            pretrain-0: T; 1,2-I
            finetune: 0-T T; 1-I I; 2-I T
        '''
        v3 = torch.zeros((len(task_mode), self.dim), device=entity_id.device)   # bsz, 200
        if finetune and not is_head:    # finetune & tail -- all T
            v3 = v2_t.clone()
            loss = 0
        else:
            v3[task_mode==0, :] = v2_t[task_mode==0, :].clone()     # T
            v3[task_mode==1, :] = self.encoder_combined_linear(torch.cat((v2_t[task_mode==1, :], v2_i[task_mode==1, :]), 1)).clone()  # I

            v4_t = self.decoder_text_linear1(v3)
            v4_i = self.decoder_visual_linear1(v3)

            v5_t = self.decoder_text_linear2(v4_t)
            v5_i = self.decoder_visual_linear2(v4_i)
            loss = self.criterion(v1_t[task_mode==0, :], v5_t[task_mode==0, :]) + self.criterion(v1_i[task_mode==1, :], v5_i[task_mode==1, :])
        '''natural'''
        return v3, loss


class TransE(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, finetune=False):
        super(TransE, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.finetune = finetune

        self.tail_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.ent_embeddings = IMG_Encoder(dim = self.dim, margin = self.margin, epsilon = self.epsilon, hidden_text_dim=self.dim)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.tail_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor = self.ent_embeddings.weight.data, 
                a = -self.embedding_range.item(), 
                b = self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.rel_embeddings.weight.data, 
                a= -self.embedding_range.item(), 
                b= self.embedding_range.item()
            )

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False


    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).squeeze(-1)
        return score

    def forward(self, data):
        if not self.finetune:        
            batch_h = data['batch_h']
            batch_t = data['batch_t']
            batch_r = data['batch_r']
            mode = data['mode']
            task_mode = data['task_mode']
            
            if mode == 'head_batch' or mode == 'normal':   # head more
                h, hloss = self.ent_embeddings(batch_h, task_mode) # combine v and t
                t = self.tail_embeddings(batch_t)
                r = self.rel_embeddings(batch_r)    
                score = self._calc(h ,t, r, mode).clone()   # bsz
                score[task_mode==1] = score[task_mode==1] + hloss
            elif mode == 'tail_batch':
                t, tloss = self.ent_embeddings(batch_t, task_mode) # combine v and t
                h = self.tail_embeddings(batch_h)
                r = self.rel_embeddings(batch_r)
                score = self._calc(h ,t, r, mode).clone()   # 11292
                score[task_mode==1] = score[task_mode==1] + tloss
            else:
                raise ValueError('Error!', mode)
            
            if self.margin_flag:
                return self.margin - score
            else:
                return score
        else:
            # finetune
            e_head = data['e_head']
            e_tail = data['e_tail']
            q_head = data['q_head']
            q_tail = data['q_tail']
            relation = data['relation']
            task_mode = data['task_mode']
            mode = data['mode']
            '''1. triple classification -> (e_head, ?, e_tail)'''
            bsz = e_head.shape[0]
            h_e_head, eh_loss = self.ent_embeddings(e_head, task_mode, finetune=True, is_head=True)     # bsz, 200
            h_e_tail, et_loss = self.ent_embeddings(e_tail, task_mode, finetune=True, is_head=False)    # bsz, 200
            h_rel = self.rel_embeddings(torch.arange(self.rel_tot).cuda())                                     # 192, 200
            rel_score = self._calc(
                            h_e_head.unsqueeze(1),                                                      # bsz, 200
                            h_e_tail.unsqueeze(1),                                                      # bsz, 200
                            h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),                     # bsz, 192, 200
                            mode
                        )   # bsz, 192
            rel_score = rel_score.argmax(dim=-1)    # bsz
            h_rel = self.rel_embeddings(rel_score)  # bsz, 200
            '''2. link prediction -> (q_head, r, ?)'''
            h_q_head, qh_loss = self.ent_embeddings(q_head, task_mode, finetune=True, is_head=True)
            h_tail = self.tail_embeddings(torch.arange(self.ent_tot).cuda())
            ent_score = self._calc(
                            h_q_head.unsqueeze(1),                                                      # bsz, 1, 200
                            h_tail.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),                    # 11292, 200
                            h_rel.unsqueeze(1),                                                         # bsz, 1, 200
                            mode
                        )   # bsz, 11292
            loss = nn.CrossEntropyLoss()(ent_score, q_tail)
            return loss, ent_score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        if not self.finetune:
            score = self.forward(data)

            # score = score + self.plus
            if self.margin_flag:
                score = self.margin - score
                return score.cpu().data.numpy()
            else:
                return score.cpu().data.numpy()
        else:
            _, ent_score = self.forward(data)
            return ent_score.cpu().data

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

class Analogy(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, finetune=False):
        super(Analogy, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.finetune = finetune

        self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim * 2)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim * 2)
  
        self.ent_img_embeddings = IMG_Encoder(dim = self.dim * 2, hidden_text_dim=self.dim*2, margin = self.margin, epsilon = self.epsilon)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
            nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor = self.ent_img_embeddings.weight.data, 
                a = -self.embedding_range.item(), 
                b = self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.rel_embeddings.weight.data, 
                a= -self.embedding_range.item(), 
                b= self.embedding_range.item()
            )

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False


    def _calc(self, h_re, h_im, h, t_re, t_im, t, r_re, r_im, r):
        return (-torch.sum(r_re * h_re * t_re +
						   r_re * h_im * t_im +
						   r_im * h_re * t_im -
						   r_im * h_im * t_re, -1)
				-torch.sum(h * t * r, -1))

    def forward(self, data):
        if not self.finetune:        
            batch_h = data['batch_h']
            batch_t = data['batch_t']
            batch_r = data['batch_r']
            mode = data['mode']
            task_mode = data['task_mode']
            if mode == 'head_batch' or mode == 'normal':   # head more
                h_re = self.ent_re_embeddings(batch_h)
                h_im = self.ent_im_embeddings(batch_h)
                h, hloss = self.ent_img_embeddings(batch_h, task_mode) # combine v and t
                t_re = self.ent_re_embeddings(batch_t)
                t_im = self.ent_im_embeddings(batch_t)
                t = self.ent_embeddings(batch_t)
                r_re = self.rel_re_embeddings(batch_r)
                r_im = self.rel_im_embeddings(batch_r)
                r = self.rel_embeddings(batch_r)    
                score = -self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r).clone()   # bsz
                score[task_mode==1] = score[task_mode==1] + hloss
            elif mode == 'tail_batch':
                h_re = self.ent_re_embeddings(batch_h)
                h_im = self.ent_im_embeddings(batch_h)
                h = self.ent_embeddings(batch_h) # combine v and t
                t_re = self.ent_re_embeddings(batch_t)
                t_im = self.ent_im_embeddings(batch_t)
                t, tloss = self.ent_img_embeddings(batch_t, task_mode)
                r_re = self.rel_re_embeddings(batch_r)
                r_im = self.rel_im_embeddings(batch_r)
                r = self.rel_embeddings(batch_r)    
                score = -self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r).clone()   # bsz
                score[task_mode==1] = score[task_mode==1] + tloss
            else:
                raise ValueError('Error!', mode)
            
            return score
        else:
            # finetune
            e_head = data['e_head']
            e_tail = data['e_tail']
            q_head = data['q_head']
            q_tail = data['q_tail']
            relation = data['relation']
            task_mode = data['task_mode']
            mode = data['mode']
            '''1. triple classification -> (e_head, ?, e_tail)'''
            bsz = e_head.shape[0]
            h_e_head_re = self.ent_re_embeddings(e_head)                                                # bsz, 200
            h_e_head_im = self.ent_im_embeddings(e_head)                                                # bsz, 200
            h_e_head, eh_loss = self.ent_img_embeddings(e_head, task_mode, finetune=True, is_head=True) # bsz, 200
            h_e_tail_re = self.ent_re_embeddings(e_tail)                                                # bsz, 200
            h_e_tail_im = self.ent_im_embeddings(e_tail)                                                # bsz, 200
            h_e_tail, et_loss = self.ent_img_embeddings(e_tail, task_mode, finetune=True, is_head=False)# bsz, 200
            h_rel_re = self.rel_re_embeddings(torch.arange(self.rel_tot).cuda())                        # 192, 200
            h_rel_im = self.rel_im_embeddings(torch.arange(self.rel_tot).cuda())                        # 192, 200
            h_rel = self.rel_embeddings(torch.arange(self.rel_tot).cuda())                              # 192, 200
            rel_score = -self._calc(
                            h_e_head_re.unsqueeze(1),
                            h_e_head_im.unsqueeze(1),
                            h_e_head.unsqueeze(1),                                      # bsz, 1. 200
                            h_e_tail_re.unsqueeze(1),
                            h_e_tail_im.unsqueeze(1),
                            h_e_tail.unsqueeze(1),                                      # bsz, 1, 200
                            h_rel_re.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),
                            h_rel_im.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),  # bsz, 192, 200
                            h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim*2),   # bsz, 192, 400
                        )   # bsz, 192

            rel_score = rel_score.argmax(dim=-1)    # bsz
            h_rel_re = self.rel_re_embeddings(rel_score)
            h_rel_im = self.rel_im_embeddings(rel_score)
            h_rel = self.rel_embeddings(rel_score)  # bsz, 200
            '''2. link prediction -> (q_head, r, ?)'''
            h_q_head_re = self.ent_re_embeddings(q_head) 
            h_q_head_im = self.ent_im_embeddings(q_head) 
            h_q_head, qh_loss = self.ent_img_embeddings(q_head, task_mode, finetune=True, is_head=True)
            h_tail_re = self.ent_re_embeddings(torch.arange(self.ent_tot).cuda())
            h_tail_im = self.ent_im_embeddings(torch.arange(self.ent_tot).cuda())
            h_tail = self.ent_embeddings(torch.arange(self.ent_tot).cuda())
            ent_score = -self._calc(
                            h_q_head_re.unsqueeze(1),
                            h_q_head_im.unsqueeze(1),
                            h_q_head.unsqueeze(1),                                                      # bsz, 1, 200
                            h_tail_re.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),                 # 11292, 200
                            h_tail_im.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),
                            h_tail.unsqueeze(0).expand(bsz, self.ent_tot, self.dim*2),                  # 11292, 400
                            h_rel_re.unsqueeze(1),
                            h_rel_im.unsqueeze(1),
                            h_rel.unsqueeze(1),                                                         # bsz, 1, 200
                        )   # bsz, 11292
            loss = nn.CrossEntropyLoss()(ent_score, q_tail)
            return loss, ent_score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        if not self.finetune:
            score = self.forward(data)
            return score.cpu().data.numpy()
        else:
            _, ent_score = self.forward(data)
            return ent_score.cpu().data

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
     

class NegativeSampling(nn.Module):

    def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
        super(NegativeSampling, self).__init__()
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        self.regul_rate = regul_rate
        self.l3_regul_rate = l3_regul_rate

    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score

    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:]
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score

    def forward(self, data):
        score = self.model(data)
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        loss_res = self.loss(p_score, n_score)
        if self.regul_rate != 0:
            loss_res += self.regul_rate * self.model.regularization(data)
        if self.l3_regul_rate != 0:
            loss_res += self.l3_regul_rate * self.model.l3_regularization()
        return loss_res


class AnalogyFinetuneDataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, data_path: str, entity2id, relation2id):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(data_path, "r") as f:
            # data in tuples (head, relation, tail)
            self.data = [line[:-1].split(" ") for line in f]

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        eh, et, eq, ea, r, m = self.data[index]

        return int(eh), int(et), int(eq), int(ea), int(r), int(m)

def create_mappings(dataset_path: str):
    """Creates separate mappings to indices for entities and relations."""
    with open(f'{dataset_path}/entity2id.txt', 'r') as f:
        lines = f.readlines()
        entity2id = {line.split('\t')[0]:line.split('\t')[1] for line in lines[1:]}
    with open(f'{dataset_path}/relation2id.txt', 'r') as f:
        lines = f.readlines()
        relation2id = {line.split('\t')[0]:line.split('\t')[1] for line in lines[1:]}
    return entity2id, relation2id

root_path = {"analogy": "data/analogy"}
img_path = {"analogy": '../MarT/dataset/MARS/images'}

root = root_path['analogy']
root_img = img_path['analogy']

entity2id, relation2id = create_mappings(dataset_path=root)

finetune=False  # ####!!!!####!!!!
analogy=False

if not finetune:
    for alpha in [1e-1]:
        print('alpha: ', alpha)
        train_dataloader = TrainDataLoader(
            in_path = "data/analogy/", 
            nbatches = 100,
            threads = 8, 
            sampling_mode = "normal", 
            bern_flag = 1, 
            filter_flag = 1, 
            neg_ent = 25,
            neg_rel = 25)

        print(train_dataloader.get_batch_size())

        # dataloader for test
        test_dataloader = TestDataLoader("data/analogy/", "link")

        # define the model
        if not analogy:
            transe = TransE(
                ent_tot = len(entity2id),
                rel_tot = len(relation2id),
                dim = 400, 
                p_norm = 1, 
                norm_flag = True)
        else:
            transe = Analogy(
                ent_tot = len(entity2id),
                rel_tot = len(relation2id),
                dim = 400, 
                p_norm = 1, 
                norm_flag = True)

        model = NegativeSampling(
            model = transe, 
            loss = MarginLoss(margin = 5.0),
            batch_size = train_dataloader.get_batch_size()
        )

        trainer = Trainer(model=model, data_loader=train_dataloader, train_times=2000, alpha=alpha, use_gpu=True)
        trainer.run()
        transe.save_checkpoint('ckpt/analogy/pt_trainse.ckpt')

        # test the model
        transe.load_checkpoint('ckpt/analogy/pt_trainse.ckpt')
        tester = Tester(model=transe, data_loader= test_dataloader, use_gpu = True)
        tester.run_link_prediction(type_constrain = False)

else:
    for alpha in [1.0]:
        print('alpha: ', alpha)
        # finetune
        bsz = 128
        train_data = AnalogyFinetuneDataset('data/analogy/train2id_ft.txt', entity2id, relation2id)
        valid_data = AnalogyFinetuneDataset('data/analogy/valid2id_ft.txt', entity2id, relation2id)
        test_data = AnalogyFinetuneDataset('data/analogy/test2id_ft.txt', entity2id, relation2id)
        
        train_dataloader = DataLoader(train_data, batch_size=bsz, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(valid_data, batch_size=bsz, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_data, batch_size=bsz, shuffle=False, num_workers=4)
        
        if not analogy:
            model = TransE(
                ent_tot=len(entity2id),
                rel_tot=len(relation2id),
                dim=200, 
                p_norm=1, 
                norm_flag=True,
                finetune=True,
            )
        else:
            model = Analogy(
                ent_tot=len(entity2id),
                rel_tot=len(relation2id),
                dim=200, 
                p_norm=1, 
                norm_flag=True,
                finetune=True,
            )
        # load pretrain ckpt
        model.load_checkpoint('ckpt/analogy/pt_transe.ckpt')
        
        trainer = Trainer(
            model=model,
            data_loader=train_dataloader,
            train_times=1000,
            alpha=alpha, 
            use_gpu=True,
            finetune=True,
        )
        trainer.run()
        model.save_checkpoint('ckpt/analogy/ft_transe.ckpt')
        
        # test the model
        model.load_checkpoint('ckpt/analogy/ft_analogy_model.ckpt')
        tester = Tester(model=model, data_loader=test_dataloader, use_gpu=True)
        mrr, mr, hit10, hti5, hti3, hit1 = tester.run_analogical_reasoning(type_constrain=False)
        print("mrr: ", mrr)
        print("mr: ", mr)
        print("hit10: ", hit10)
        print("hti5: ", hti5)
        print("hti3: ", hti3)
        print("hit1: ", hit1)