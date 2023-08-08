import torch
import os
import re
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils import data
import torch.optim as optim
from tqdm import tqdm
import ctypes
import pickle
from DATA_ import TrainDataLoader, TestDataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
        for data in self.data_loader:
            predictions = self.test_one_step_ft(data)   # bsz, 11292
            truth = data[3]
            _, predictions = torch.sort(predictions, dim=1, descending=True)
            _, predictions = torch.sort(predictions, dim=1)
            rank = predictions[torch.arange(truth.shape[0]), truth].detach().cpu() + 1
            ranks.append(rank)
            
        ranks = torch.cat(ranks).float()
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


class TransE(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, finetune=False):
        super(TransE, self).__init__()
        with open('data/analogy/entity2id.txt') as fp:
            entity2id = fp.readlines()[1:]
            entity2id = {line.split('\t')[0]: line.split('\t')[1] for line in entity2id}

        self.entity2id = entity2id
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.finetune = finetune

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.ent_project_layer = nn.Linear(self.dim, self.dim)
        
        self.visual_embedding = self._init_visual_emb()
        self.img_project_layer = nn.Linear(4096, self.dim)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False
            
    def _init_visual_emb(self):
        uniform_range = 6 / np.sqrt(self.dim)
        weights = torch.empty(self.ent_tot + 1, 4096)
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
        entities_emb = nn.Embedding.from_pretrained(weights, padding_idx=self.ent_tot)
        return entities_emb


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
            head_ent = self.ent_embeddings(batch_h) # bsz, dim
            tail_ent = self.ent_embeddings(batch_t)
            r = self.rel_embeddings(batch_r)
            head_ent = self.ent_project_layer(head_ent)
            tail_ent = self.ent_project_layer(tail_ent)
            
            # calc image_attn_vec with attention
            head_attn_vec = self.img_project_layer(self.visual_embedding(batch_h))   # bsz, n, 50
            tail_attn_vec = self.img_project_layer(self.visual_embedding(batch_t))
            
            # cal tt
            tt_score = self._calc(head_ent, tail_ent, r, mode)
            
            # cal ii
            ii_score = self._calc(head_attn_vec, tail_attn_vec, r, mode)
            
            # cal ti
            ti_score = self._calc(head_ent, tail_attn_vec, r, mode)
            
            # cal it
            it_score = self._calc(head_attn_vec, tail_ent, r, mode)
            
            # task_mode=0: (T,T), task_mode=1: (I,T), task_mode=2: (I,I)
            tt_idx = task_mode == 0
            it_idx = task_mode == 1
            ii_idx = task_mode == 2
            score = torch.zeros_like(tt_score, device=tt_score.device)
            score[tt_idx] += tt_score[tt_idx].clone()
            score[it_idx] += it_score[it_idx].clone() + ti_score[it_idx].clone()
            score[ii_idx] += ii_score[ii_idx].clone()


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
            h_e_head = self.ent_embeddings(e_head)     # bsz, 200
            h_e_tail = self.ent_embeddings(e_tail)    # bsz, 200
            h_rel = self.rel_embeddings(torch.arange(self.rel_tot).cuda())                                     # 192, 200
            # image vec
            e_head_attn_vec = self.img_project_layer(self.visual_embedding(e_head))   # bsz, n, 50
            e_tail_attn_vec = self.img_project_layer(self.visual_embedding(e_tail))
            # cal tt
            tt_score = self._calc(h_e_head.unsqueeze(1), h_e_tail.unsqueeze(1), h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim), mode)
            # cal ii
            ii_score = self._calc(e_head_attn_vec.unsqueeze(1), e_tail_attn_vec.unsqueeze(1), h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim), mode)
            # cal ti
            ti_score = self._calc(h_e_head.unsqueeze(1), e_tail_attn_vec.unsqueeze(1), h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim), mode)
            # cal it
            it_score = self._calc(e_head_attn_vec.unsqueeze(1), h_e_tail.unsqueeze(1), h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim), mode)
            # 0: (T,T), 1: (I, I), 2:(I, T)
            tt_idx = task_mode == 0
            it_idx = task_mode == 2
            ii_idx = task_mode == 1
            rel_score = torch.zeros_like(tt_score, device=tt_score.device)
            rel_score[tt_idx] += tt_score[tt_idx].clone()
            rel_score[it_idx] += it_score[it_idx].clone() + ti_score[it_idx].clone()
            rel_score[ii_idx] += ii_score[ii_idx].clone()
            rel_score = rel_score.argmax(dim=-1)    # bsz
            h_rel = self.rel_embeddings(rel_score)  # bsz, 200
            '''2. link prediction -> (q_head, r, ?)'''
            h_q_head = self.ent_embeddings(q_head)
            q_head_attn_vec = self.img_project_layer(self.visual_embedding(q_head))   # bsz, n, 50
            h_tail = self.ent_embeddings(torch.arange(self.ent_tot).cuda())
            h_tail_attn_vec = self.img_project_layer(self.visual_embedding(torch.arange(self.ent_tot).cuda()))   # bsz, n, 50
            # cal tt
            tt_score = self._calc(h_q_head.unsqueeze(1), h_tail.unsqueeze(0).expand(bsz, self.ent_tot, self.dim), h_rel.unsqueeze(1), mode)
            # cal ii
            ii_score = self._calc(q_head_attn_vec.unsqueeze(1), h_tail_attn_vec.unsqueeze(0).expand(bsz, self.ent_tot, self.dim), h_rel.unsqueeze(1), mode)
            # cal ti
            ti_score = self._calc(h_q_head.unsqueeze(1), h_tail_attn_vec.unsqueeze(0).expand(bsz, self.ent_tot, self.dim), h_rel.unsqueeze(1), mode)
            # cal it
            it_score = self._calc(q_head_attn_vec.unsqueeze(1), h_tail.unsqueeze(0).expand(bsz, self.ent_tot, self.dim), h_rel.unsqueeze(1), mode)
            # 0: (T, T), 1: (I, I), 2: (I, T)
            tt_idx = task_mode == 0
            it_idx = task_mode == 2
            ii_idx = task_mode == 1
            ent_score = torch.zeros_like(tt_score, device=tt_score.device)
            ent_score[tt_idx] += tt_score[tt_idx].clone()
            ent_score[it_idx] += it_score[it_idx].clone() + ti_score[it_idx].clone()
            ent_score[ii_idx] += ii_score[ii_idx].clone()   # bsz, 11292
            
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
        with open('data/analogy/entity2id.txt') as fp:
            entity2id = fp.readlines()[1:]
            entity2id = {line.split('\t')[0]: line.split('\t')[1] for line in entity2id}

        self.entity2id = entity2id
        
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
  
        self.visual_embedding = self._init_visual_emb()
        self.img_project_layer = nn.Linear(4096, self.dim*2)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
            nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.img_project_layer.weight.data)

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False
            
    def _init_visual_emb(self):
        uniform_range = 6 / np.sqrt(self.dim)
        weights = torch.empty(self.ent_tot + 1, 4096)
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
        # weights = F.layer_norm(weights, weights.shape)
        entities_emb = nn.Embedding.from_pretrained(weights, padding_idx=self.ent_tot)
        return entities_emb


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
            h_re = self.ent_re_embeddings(batch_h)
            h_im = self.ent_im_embeddings(batch_h)  # bsz, 200
            h = self.ent_embeddings(batch_h)        # bsz, 400
            t_re = self.ent_re_embeddings(batch_t)
            t_im = self.ent_im_embeddings(batch_t)
            t = self.ent_embeddings(batch_t)
            r_re = self.rel_re_embeddings(batch_r)
            r_im = self.rel_im_embeddings(batch_r)
            r = self.rel_embeddings(batch_r)    
            
            # calc image_attn_vec with attention
            head_attn_vec = self.img_project_layer(self.visual_embedding(batch_h))   # bsz, n, 50
            tail_attn_vec = self.img_project_layer(self.visual_embedding(batch_t))
            # cal tt
            tt_score = self._calc(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)
            # cal ii
            ii_score = self._calc(h_re, h_im, head_attn_vec, t_re, t_im, tail_attn_vec, r_re, r_im, r)
            # cal ti
            ti_score = self._calc(h_re, h_im, h, t_re, t_im, tail_attn_vec, r_re, r_im, r)
            # cal it
            it_score = self._calc(h_re, h_im, head_attn_vec, t_re, t_im, t, r_re, r_im, r)
            
            # task_mode=0: (T,T), task_mode=1: (I,T), task_mode=2: (I,I)
            tt_idx = task_mode == 0
            it_idx = task_mode == 1
            ii_idx = task_mode == 2
            score = torch.zeros_like(tt_score, device=tt_score.device)
            score[tt_idx] += tt_score[tt_idx].clone()
            score[it_idx] += it_score[it_idx].clone() + ti_score[it_idx].clone()
            score[ii_idx] += ii_score[ii_idx].clone()
                  
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
            h_e_head = self.ent_embeddings(e_head)
            h_e_tail_re = self.ent_re_embeddings(e_tail)                                                # bsz, 200
            h_e_tail_im = self.ent_im_embeddings(e_tail)                                                # bsz, 200
            h_e_tail = self.ent_embeddings(e_tail)
            h_rel_re = self.rel_re_embeddings(torch.arange(self.rel_tot).cuda())                        # 192, 200
            h_rel_im = self.rel_im_embeddings(torch.arange(self.rel_tot).cuda())                        # 192, 200
            h_rel = self.rel_embeddings(torch.arange(self.rel_tot).cuda())                              # 192, 200
            
            # calc image_attn_vec with attention
            head_attn_vec = self.img_project_layer(self.visual_embedding(e_head))   # bsz, n, 50
            tail_attn_vec = self.img_project_layer(self.visual_embedding(e_tail))
            # cal tt
            tt_score = -self._calc(
                            h_e_head_re.unsqueeze(1),
                            h_e_head_im.unsqueeze(1),
                            h_e_head.unsqueeze(1),                                      # bsz, 1. 200
                            h_e_tail_re.unsqueeze(1),
                            h_e_tail_im.unsqueeze(1),
                            h_e_tail.unsqueeze(1),                                      # bsz, 1, 200
                            h_rel_re.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),
                            h_rel_im.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),  # bsz, 192, 200
                            h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim*2),   # bsz, 192, 400
                        )   # bsz, 192)
            # cal ii
            ii_score = -self._calc(
                            h_e_head_re.unsqueeze(1),
                            h_e_head_im.unsqueeze(1),
                            head_attn_vec.unsqueeze(1),                                      # bsz, 1. 200
                            h_e_tail_re.unsqueeze(1),
                            h_e_tail_im.unsqueeze(1),
                            tail_attn_vec.unsqueeze(1),                                      # bsz, 1, 200
                            h_rel_re.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),
                            h_rel_im.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),  # bsz, 192, 200
                            h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim*2),   # bsz, 192, 400
                        )   # bsz, 192)            
            # cal ti
            ti_score = -self._calc(
                            h_e_head_re.unsqueeze(1),
                            h_e_head_im.unsqueeze(1),
                            h_e_head.unsqueeze(1),                                      # bsz, 1. 200
                            h_e_tail_re.unsqueeze(1),
                            h_e_tail_im.unsqueeze(1),
                            tail_attn_vec.unsqueeze(1),                                      # bsz, 1, 200
                            h_rel_re.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),
                            h_rel_im.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),  # bsz, 192, 200
                            h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim*2),   # bsz, 192, 400
                        )   # bsz, 192)
            # cal it
            it_score = -self._calc(
                            h_e_head_re.unsqueeze(1),
                            h_e_head_im.unsqueeze(1),
                            head_attn_vec.unsqueeze(1),                                      # bsz, 1. 200
                            h_e_tail_re.unsqueeze(1),
                            h_e_tail_im.unsqueeze(1),
                            h_e_tail.unsqueeze(1),                                      # bsz, 1, 200
                            h_rel_re.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),
                            h_rel_im.unsqueeze(0).expand(bsz, self.rel_tot, self.dim),  # bsz, 192, 200
                            h_rel.unsqueeze(0).expand(bsz, self.rel_tot, self.dim*2),   # bsz, 192, 400
                        )   # bsz, 192)        
   
            # 0: (T,T), 1: (I, I), 2:(I, T)
            tt_idx = task_mode == 0
            it_idx = task_mode == 2
            ii_idx = task_mode == 1
            rel_score = torch.zeros_like(tt_score, device=tt_score.device)
            rel_score[tt_idx] += tt_score[tt_idx].clone()
            rel_score[it_idx] += it_score[it_idx].clone() + ti_score[it_idx].clone()
            rel_score[ii_idx] += ii_score[ii_idx].clone()
            rel_score = rel_score.argmax(dim=-1)    # bsz
            h_rel_re = self.rel_re_embeddings(rel_score)
            h_rel_im = self.rel_im_embeddings(rel_score)
            h_rel = self.rel_embeddings(rel_score)  # bsz, 200
            '''2. link prediction -> (q_head, r, ?)'''
            h_q_head_re = self.ent_re_embeddings(q_head) 
            h_q_head_im = self.ent_im_embeddings(q_head) 
            h_q_head = self.ent_embeddings(q_head)
            h_tail_re = self.ent_re_embeddings(torch.arange(self.ent_tot).cuda())
            h_tail_im = self.ent_im_embeddings(torch.arange(self.ent_tot).cuda())
            h_tail = self.ent_embeddings(torch.arange(self.ent_tot).cuda())
            # image vec
            q_head_attn_vec = self.img_project_layer(self.visual_embedding(q_head))   # bsz, n, 50
            q_tail_attn_vec = self.img_project_layer(self.visual_embedding(torch.arange(self.ent_tot).cuda()))
            # cal tt
            tt_score = -self._calc(
                            h_q_head_re.unsqueeze(1),
                            h_q_head_im.unsqueeze(1),
                            h_q_head.unsqueeze(1),                                                      # bsz, 1, 200
                            h_tail_re.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),                 # 11292, 200
                            h_tail_im.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),
                            h_tail.unsqueeze(0).expand(bsz, self.ent_tot, self.dim*2),                  # 11292, 400
                            h_rel_re.unsqueeze(1),
                            h_rel_im.unsqueeze(1),
                            h_rel.unsqueeze(1),  
                        )   # bsz, 192)
            # cal ii
            ii_score = -self._calc(
                            h_q_head_re.unsqueeze(1),
                            h_q_head_im.unsqueeze(1),
                            q_head_attn_vec.unsqueeze(1),                                                      # bsz, 1, 200
                            h_tail_re.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),                 # 11292, 200
                            h_tail_im.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),
                            q_tail_attn_vec.unsqueeze(0).expand(bsz, self.ent_tot, self.dim*2),                  # 11292, 400
                            h_rel_re.unsqueeze(1),
                            h_rel_im.unsqueeze(1),
                            h_rel.unsqueeze(1),  
                        )   # bsz, 192)            
            # cal ti
            ti_score = -self._calc(
                            h_q_head_re.unsqueeze(1),
                            h_q_head_im.unsqueeze(1),
                            h_q_head.unsqueeze(1),                                                      # bsz, 1, 200
                            h_tail_re.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),                 # 11292, 200
                            h_tail_im.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),
                            q_tail_attn_vec.unsqueeze(0).expand(bsz, self.ent_tot, self.dim*2),                  # 11292, 400
                            h_rel_re.unsqueeze(1),
                            h_rel_im.unsqueeze(1),
                            h_rel.unsqueeze(1),  
                        )   # bsz, 192)
            # cal it
            it_score = -self._calc(
                            h_q_head_re.unsqueeze(1),
                            h_q_head_im.unsqueeze(1),
                            q_head_attn_vec.unsqueeze(1),                                                      # bsz, 1, 200
                            h_tail_re.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),                 # 11292, 200
                            h_tail_im.unsqueeze(0).expand(bsz, self.ent_tot, self.dim),
                            q_tail_attn_vec.unsqueeze(0).expand(bsz, self.ent_tot, self.dim*2),                  # 11292, 400
                            h_rel_re.unsqueeze(1),
                            h_rel_im.unsqueeze(1),
                            h_rel.unsqueeze(1),  
                        )   # bsz, 192)       
            
            # 0: (T, T), 1: (I, I), 2: (I, T)
            tt_idx = task_mode == 0
            it_idx = task_mode == 2
            ii_idx = task_mode == 1
            ent_score = torch.zeros_like(tt_score, device=tt_score.device)
            ent_score[tt_idx] += tt_score[tt_idx].clone()
            ent_score[it_idx] += it_score[it_idx].clone() + ti_score[it_idx].clone()
            ent_score[ii_idx] += ii_score[ii_idx].clone()   # bsz, 11292

            loss = nn.CrossEntropyLoss()(ent_score, q_tail)
            return loss, ent_score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)  # bsz, 200
        h = self.ent_embeddings(batch_h)        # bsz, 400
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        t = self.ent_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        r = self.rel_embeddings(batch_r)    
        regul = (torch.mean(h_re ** 2) + 
				 torch.mean(h_im ** 2) + 
				 torch.mean(h ** 2) + 
				 torch.mean(t_re ** 2) + 
				 torch.mean(t_im ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r_re ** 2) + 
				 torch.mean(r_im ** 2) + 
				 torch.mean(r ** 2)) / 9
        return regul

    def predict(self, data):
        if not self.finetune:
            score = -self.forward(data)
            return score.cpu().data.numpy()
        else:
            _, ent_score = self.forward(data)
            return ent_score.cpu().data

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)), strict=False)
        self.eval()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
     
     
class SoftplusLoss(nn.Module):
    
	def __init__(self, adv_temperature = None):
		super(SoftplusLoss, self).__init__()
		self.criterion = nn.Softplus()
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return (self.criterion(-p_score).mean() + (self.get_weights(n_score) * self.criterion(n_score)).sum(dim = -1).mean()) / 2
		else:
			return (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2
			

	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()

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
img_path = {"analogy": '../AnalogyKG/dataset/MARS/images'}

root = root_path['analogy']
root_img = img_path['analogy']

entity2id, relation2id = create_mappings(dataset_path=root)

finetune = False  # ####!!!!####!!!!
analogy = False

if not finetune:
    for alpha in [1.0]:
        print('alpha: ', alpha)
        if not analogy:
            # transe
            train_dataloader = TrainDataLoader(
                in_path = "data/analogy/", 
                nbatches = 100,
                threads = 8, 
                sampling_mode = "normal", 
                bern_flag = 1, 
                filter_flag = 1, 
                neg_ent = 25,
                neg_rel = 25)
            
            # define the model
            transe = TransE(
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
        else:
            # analogy
            train_dataloader = TrainDataLoader(
                in_path = "data/analogy/", 
                nbatches = 100,
                threads = 8, 
                sampling_mode = "normal", 
                bern_flag = 1, 
                filter_flag = 1, 
                neg_ent = 25,
                neg_rel = 25)

            # define the model
            transe = Analogy(
                ent_tot = len(entity2id),
                rel_tot = len(relation2id),
                dim = 200, 
                p_norm = 1, 
                norm_flag = True)
            
            model = NegativeSampling(
                model = transe, 
                loss = SoftplusLoss(),
                batch_size = train_dataloader.get_batch_size(),
                regul_rate = 1.0
            )

        print(train_dataloader.get_batch_size())

        # dataloader for test
        test_dataloader = TestDataLoader("data/analogy/", "link")

        trainer = Trainer(model=model, data_loader=train_dataloader, train_times=2000, alpha=alpha, use_gpu=True)
        trainer.run()
        transe.save_checkpoint('ckpt/analogy/pt_ikrl_transe.ckpt')

        # test the model
        transe.load_checkpoint('ckpt/analogy/pt_ikrl_transe.ckpt')
        tester = Tester(model=transe, data_loader= test_dataloader, use_gpu = True)
        tester.run_link_prediction(type_constrain = False)

else:
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
    model.load_checkpoint('ckpt/analogy/pt_ikrl_transe.ckpt')
    
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=1000,
        alpha=1e-4,
        use_gpu=True,
        finetune=True,
        opt_method='Adam',
    )
    trainer.run()
    model.save_checkpoint('ckpt/analogy/ft_ikrl_model.ckpt')
    
    # test the model
    model.load_checkpoint('ckpt/analogy/ft_ikrl_model.ckpt')
    tester = Tester(model=model, data_loader=test_dataloader, use_gpu=True)
    mrr, mr, hit10, hit5, hti3, hit1 = tester.run_analogical_reasoning(type_constrain=False)
    print("mrr: ", mrr)
    print("mr: ", mr)
    print("hit10: ", hit10)
    print("hti5: ", hit5)
    print("hti3: ", hti3)
    print("hit1: ", hit1)
    
    