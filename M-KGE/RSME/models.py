from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
import pickle
import torch.nn.functional as F
import numpy as np
from config import alpha,beta,random_gate,forget_gate,remember_rate,constant


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """

        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0

                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    if not constant:
                        r_embeddings, img_embeddings= self.get_rhs(c_begin, chunk_size)
                        h_r = self.get_queries(these_queries)
                        n = len(h_r)
                        scores_str = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

                        for i in range(n):
                            i_alpha = self.alpha[(these_queries[i, 1])]
                            single_score = h_r[[i], :] @ (
                                    (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0,
                                                                                                                      1)
                            scores_str = torch.cat((scores_str, single_score.detach()), 0)
                    else:
                        rhs = self.get_rhs(c_begin, chunk_size) # 2000, 10182
                        q = self.get_queries(these_queries)     # bsz, 2000
                        scores_str = q @ rhs                    # bsz, 10182

                    lhs_img = F.normalize(self.img_vec[these_queries[:,0]], p=2, dim=1) # bsz, 1000
                    rhs_img = F.normalize(self.img_vec, p=2, dim=1).transpose(0, 1)     # bsz, 10182
                    score_img = lhs_img @ rhs_img                                       # bsz, 500, 10182
                    # beta=0.95
                    if forget_gate:
                        scores = torch.zeros_like(score_img, device=score_img.device)   # bsz, 10182
                        for i in range(len(these_queries)):
                            mode = queries[i, -1].item()
                            if mode == 0:   #（T， T）
                                scores[i] = scores_str[i]
                            elif mode == 1: # (I, T)
                                scores[i] = beta*scores_str[i]
                            else:           # (I, I)
                                scores[i] = beta*scores_str[i] + (1-beta)*score_img[i]*self.rel_pd[these_queries[i, 1]]
                    else:
                        scores = beta * scores_str + (1 - beta) * score_img
                    targets = self.score(these_queries) # bsz, 1
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data

def sc_wz_01(len,num_1):
    A=[1 for i in range(num_1)]
    B=[0 for i in range(len-num_1)]
    C=A+B
    np.random.shuffle(C)
    return np.array(C,dtype=np.float)

def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        return [line[:-1] for line in lines]

class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,
            finetune: bool = False,
            img_info='data/analogy/img_vec_id_analogy_vit.pickle',
            sig_alpha='data/analogy/rel_MPR_SIG_vit.pickle',
            rel_pd='data/analogy/rel_MPR_PD_vit_mrp{}.pickle'
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.finetune = finetune
        self.analogy_entids = read_txt('data/analogy/analogy_ent_id')
        self.analogy_relids = read_txt('data/analogy/analogy_rel_id')

        self.r_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])

        self.r_embeddings[0].weight.data *= init_size
        self.r_embeddings[1].weight.data *= init_size
        if not constant:
            self.alpha=torch.from_numpy(pickle.load(open(sig_alpha, 'rb'))).cuda()
            self.alpha=torch.cat((self.alpha,self.alpha),dim=0)
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)  # [14951, 2000]
        self.img_dimension = 1000
        self.img_info = pickle.load(open(img_info, 'rb'))
        self.img_vec = torch.from_numpy(self.img_info).float().cuda()
        if not random_gate:
            self.rel_pd=torch.from_numpy(pickle.load(open(rel_pd.format(remember_rate),'rb'))).cuda()
        else:
            tmp=pickle.load(open(rel_pd.format(remember_rate), 'rb'))
            self.rel_pd=torch.from_numpy(sc_wz_01(len(tmp),np.sum(tmp))).unsqueeze(1).cuda()

        self.rel_pd=torch.cat((self.rel_pd,self.rel_pd),dim=0)
        # self.alpha[self.img_info['missed'], :] = 1

        self.post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform(self.post_mats)

    def score(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 2])]

            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1)

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str = torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            return beta * score_str + (1 - beta) * score_img
        else:
            rel = self.r_embeddings[1](x[:, 1])

            lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
            for i in range(len(x)):
                mode = x[i, -1].item()
                if mode == 0:   # （T， T）
                    lhs[i] = self.r_embeddings[0](x[i, 0])
                    rhs[i] = self.r_embeddings[0](x[i, 2])
                elif mode == 1: # (I, T)
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs[i] = self.r_embeddings[0](x[i, 2])
                else:           # (I, I)
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
     
            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img,1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img,rhs_img, 1).unsqueeze(1)


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str=torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            for i in range(len(score_str)):
                mode = x[i, -1].item()
                if mode == 0:   # （T， T）
                    continue
                elif mode == 1: # (I, T)
                    continue
                else:           # (I, I)
                    score_str[i] = beta * score_str[i] + (1-beta) * score_img[i]
            return score_str


    def forward(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 2])]

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            h_r = torch.cat((lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]), dim=-1)

            n = len(h_r)
            ans = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

            for i in range(n):
                i_alpha = self.alpha[(x[i, 1])]
                single_score = h_r[[i], :] @ (
                            (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0, 1)
                ans = torch.cat((ans, single_score.detach()), 0)

            return (ans), (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                           torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                           torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
        else:
            if not self.finetune:
                embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings    # (entity, 2000)
                
                rel = self.r_embeddings[1](x[:, 1])
                
                lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    if mode == 0:   # （T， T）
                        lhs[i] = self.r_embeddings[0](x[i, 0])
                        rhs[i] = self.r_embeddings[0](x[i, 2])
                    elif mode == 1: # (I, T)
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = self.r_embeddings[0](x[i, 2])
                    else:           # (I, I)
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]

                lhs = lhs[:, :self.rank], lhs[:, self.rank:]    # (1000, 1000), (1000, 1000)
                rel = rel[:, :self.rank], rel[:, self.rank:]
                rhs = rhs[:, :self.rank], rhs[:, self.rank:]

                to_score = embedding
                to_score = to_score[:, :self.rank], to_score[:, self.rank:]

                return (
                            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
                    ), (
                        torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2) 
                    )
            else:
                '''analogical reasoning
                    x : [e_h, e_t, q, a, r, mode]
                    1. triple classification (e_h, ?, e_t) -> r
                    2. link prediction (a, r, ?) -> a
                '''
                # 1. triple classification
                rel = self.get_relations()    # 2000, 382
                lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
                for i in range(len(x)):
                    mode = x[i, -1].item()
                    if mode == 0:   # （T， T）
                        lhs[i] = self.r_embeddings[0](x[i, 0])
                        rhs[i] = self.r_embeddings[0](x[i, 1])
                    elif mode == 1: # (I, T)
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = self.r_embeddings[0](x[i, 1])
                    else:           # (I, I)
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 1]) + self.alpha * img_embeddings[x[i, 1]]

                lhs = lhs[:, :self.rank], lhs[:, self.rank:]
                rhs = rhs[:, :self.rank], rhs[:, self.rank:]
                
                q = torch.cat([
                        lhs[0] * rhs[0] - lhs[1] * rhs[1],
                        lhs[0] * rhs[1] + lhs[1] * rhs[0]], 1)  # bsz, 2000
                scores_r = q @ rel
                scores_r = scores_r.argmax(dim=-1)              # bsz, 1
                
                # 2. link prediction
                pred_rel = self.r_embeddings[1](scores_r)       # bsz, 2000
                a_lhs = torch.rand((len(x), 2*self.rank)).to(x.device)  # 1000, 2000, (head)
                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    if mode == 0:   # （T， T）
                        a_lhs[i] = self.r_embeddings[0](x[i, 2])
                    elif mode == 1: # (I, T)
                        a_lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                    else:           # (I, I)
                        a_lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]

                a_lhs = a_lhs[:, :self.rank], a_lhs[:, self.rank:]    # (1000, 1000), (1000, 1000)
                pred_rel = pred_rel[:, :self.rank], pred_rel[:, self.rank:]

                embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings    # (entity, 2000)
                to_score = embedding
                to_score = to_score[:, :self.rank], to_score[:, self.rank:]

                return (
                            (a_lhs[0] * pred_rel[0] - a_lhs[1] * pred_rel[1]) @ to_score[0].transpose(0, 1) +
                            (a_lhs[0] * pred_rel[1] + a_lhs[1] * pred_rel[0]) @ to_score[1].transpose(0, 1)
                    ), (
                        torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2),
                        torch.sqrt(pred_rel[0] ** 2 + pred_rel[1] ** 2),
                        torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2) 
                    )
                                     
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            return self.r_embeddings[0].weight.data[
                   chunk_begin:chunk_begin + chunk_size
                   ],img_embeddings
        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            return embedding[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)
            
    def get_relations(self):
        return self.r_embeddings[1].weight.transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(queries[:, 1])]) * self.r_embeddings[0](queries[:, 0]) + self.alpha[(queries[:, 1])] * img_embeddings[
                (queries[:, 0])]
            rel = self.r_embeddings[1](queries[:, 1])

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)
        else:
            rel = self.r_embeddings[1](queries[:, 1])
            lhs = torch.rand((len(queries), 2*self.rank)).to(queries.device)
            for i in range(len(queries)):
                mode = queries[i, -1].item()
                if mode == 0:   # （T， T）
                    lhs[i] = self.r_embeddings[0](queries[i, 0])
                elif mode == 1: # (I, T)
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                else:           # (I, I)
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)


class Analogy(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,
            finetune: bool = False,
            img_info='data/analogy/img_vec_id_analogy_vit.pickle',
            sig_alpha='data/analogy/rel_MPR_SIG_vit.pickle',
            rel_pd='data/analogy/rel_MPR_PD_vit_mrp{}.pickle'
    ):
        super(Analogy, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.finetune = finetune
        self.analogy_entids = read_txt('data/analogy/analogy_ent_id')
        self.analogy_relids = read_txt('data/analogy/analogy_rel_id')

        self.r_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])  # entity and relation
        
        self.ent_embeddings = nn.Embedding(self.sizes[0], self.rank * 2)
        self.rel_embeddings = nn.Embedding(self.sizes[1], self.rank * 2)

        self.r_embeddings[0].weight.data *= init_size
        self.r_embeddings[1].weight.data *= init_size
        self.ent_embeddings.weight.data *= init_size
        self.rel_embeddings.weight.data *= init_size
        
        if not constant:
            self.alpha=torch.from_numpy(pickle.load(open(sig_alpha, 'rb'))).cuda()
            self.alpha=torch.cat((self.alpha,self.alpha),dim=0)
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)  # [14951, 2000]
        self.img_dimension = 1000
        self.img_info = pickle.load(open(img_info, 'rb'))
        self.img_vec = torch.from_numpy(self.img_info).float().cuda()
        if not random_gate:
            self.rel_pd=torch.from_numpy(pickle.load(open(rel_pd.format(remember_rate),'rb'))).cuda()
        else:
            tmp=pickle.load(open(rel_pd.format(remember_rate), 'rb'))
            self.rel_pd=torch.from_numpy(sc_wz_01(len(tmp),np.sum(tmp))).unsqueeze(1).cuda()

        self.rel_pd=torch.cat((self.rel_pd,self.rel_pd),dim=0)

        self.post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform(self.post_mats)
        
    def score(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 2])]

            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1)

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str = torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            return beta * score_str + (1 - beta) * score_img
        else:
            rel = self.r_embeddings[1](x[:, 1])
            rel_rel = self.rel_embeddings(x[:, 1])

            lhs_ent, rhs_ent = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)
            lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
            for i in range(len(x)):
                mode = x[i, -1].item()
                if mode == 0:   # （T， T）
                    lhs_ent[i] = self.ent_embeddings(x[i, 0])
                    rhs_ent[i] = self.ent_embeddings(x[i, 2])
                    lhs[i] = self.r_embeddings[0](x[i, 0])
                    rhs[i] = self.r_embeddings[0](x[i, 2])
                elif mode == 1: # (I, T)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs_ent[i] = self.ent_embeddings(x[i, 2])
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs[i] = self.r_embeddings[0](x[i, 2])
                else:           # (I, I)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                    rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
     
            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img,1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img,rhs_img, 1).unsqueeze(1)


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            
            score_str=torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            ) + torch.sum((lhs_ent * rel_rel) * rhs_ent, 1, keepdim=True)
            # beta = 0.95
            for i in range(len(score_str)):
                mode = x[i, -1].item()
                if mode == 0:   # （T， T）
                    continue
                elif mode == 1: # (I, T)
                    continue
                else:           # (I, I)
                    score_str[i] = beta * score_str[i] + (1-beta) * score_img[i]
            return score_str

    def forward(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[(x[:, 2])]

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            h_r = torch.cat((lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]), dim=-1)

            n = len(h_r)
            ans = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

            for i in range(n):
                i_alpha = self.alpha[(x[i, 1])]
                single_score = h_r[[i], :] @ (
                            (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0, 1)
                ans = torch.cat((ans, single_score.detach()), 0)

            return (ans), (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                           torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                           torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
        else:
            if not self.finetune:
                embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings    # (entity, 2000)
                
                rel_rel = self.rel_embeddings(x[:, 1])
                rel = self.r_embeddings[1](x[:, 1])
                
                lhs_ent = torch.rand((len(x), 2*self.rank)).to(x.device)
                lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    if mode == 0:   # （T， T）
                        lhs_ent[i] = self.ent_embeddings(x[i, 0])
                        lhs[i] = self.r_embeddings[0](x[i, 0])
                        rhs[i] = self.r_embeddings[0](x[i, 2])
                    elif mode == 1: # (I, T)
                        lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = self.r_embeddings[0](x[i, 2])
                    else:           # (I, I)
                        lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]

                lhs = lhs[:, :self.rank], lhs[:, self.rank:]    # (1000, 1000), (1000, 1000)
                rel = rel[:, :self.rank], rel[:, self.rank:]
                rhs = rhs[:, :self.rank], rhs[:, self.rank:]
                
                to_score = embedding
                to_score = to_score[:, :self.rank], to_score[:, self.rank:]
                
                to_score_ent = self.ent_embeddings.weight

                return (
                            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1) +
                            (lhs_ent * rel_rel) @ to_score_ent.transpose(0, 1)
                    ), (
                        torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
                        torch.sqrt(lhs_ent[0] ** 2 + rel_rel[1] ** 2),
                    )
            else:
                '''analogical reasoning
                    x : [e_h, e_t, q, a, r, mode]
                    1. triple classification (e_h, ?, e_t) -> r
                    2. link prediction (a, r, ?) -> a
                '''
                # 1. triple classification
                rel = self.get_relations()    # 2000, 382
                rel_rel = self.rel_embeddings.weight
                
                lhs_ent, rhs_ent = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)
                lhs, rhs = torch.rand((len(x), 2*self.rank)).to(x.device), torch.rand((len(x), 2*self.rank)).to(x.device)   # 1000, 2000, (head)
                for i in range(len(x)):
                    mode = x[i, -1].item()
                    if mode == 0:   # （T， T）
                        lhs_ent[i] = self.ent_embeddings(x[i, 0])
                        rhs_ent[i] = self.ent_embeddings(x[i, 1])
                        lhs[i] = self.r_embeddings[0](x[i, 0])
                        rhs[i] = self.r_embeddings[0](x[i, 1])
                    elif mode == 1: # (I, T)
                        lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs_ent[i] = self.ent_embeddings(x[i, 1])
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = self.r_embeddings[0](x[i, 1])
                    else:           # (I, I)
                        lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 1]) + self.alpha * img_embeddings[x[i, 1]]
                        lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 0]) + self.alpha * img_embeddings[x[i, 0]]
                        rhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 1]) + self.alpha * img_embeddings[x[i, 1]]

                lhs = lhs[:, :self.rank], lhs[:, self.rank:]
                rhs = rhs[:, :self.rank], rhs[:, self.rank:]
                
                q = torch.cat([
                        lhs[0] * rhs[0] - lhs[1] * rhs[1],
                        lhs[0] * rhs[1] + lhs[1] * rhs[0]], 1)  + lhs_ent * rhs_ent
                scores_r = q @ rel
                scores_r = scores_r.argmax(dim=-1)              # bsz, 1
                
                # 2. link prediction
                pred_rel = self.r_embeddings[1](scores_r)       # bsz, 2000
                pred_rel_rel = self.rel_embeddings(scores_r)
                
                a_lhs_ent = torch.rand((len(x), 2*self.rank)).to(x.device)
                a_lhs = torch.rand((len(x), 2*self.rank)).to(x.device)  # 1000, 2000, (head)
                modes = x[:, -1].detach().cpu().tolist()
                for i in range(len(x)):
                    mode = modes[i]
                    if mode == 0:   # （T， T）
                        a_lhs_ent[i] = self.ent_embeddings(x[i, 2])
                        a_lhs[i] = self.r_embeddings[0](x[i, 2])
                    elif mode == 1: # (I, T)
                        a_lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                        a_lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                    else:           # (I, I)
                        a_lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]
                        a_lhs[i] = (1 - self.alpha) * self.r_embeddings[0](x[i, 2]) + self.alpha * img_embeddings[x[i, 2]]

                a_lhs = a_lhs[:, :self.rank], a_lhs[:, self.rank:]    # (1000, 1000), (1000, 1000)
                pred_rel = pred_rel[:, :self.rank], pred_rel[:, self.rank:]

                embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings    # (entity, 2000)
                to_score = embedding
                to_score = to_score[:, :self.rank], to_score[:, self.rank:]
                
                to_score_ent = self.ent_embeddings.weight

                return (
                            (a_lhs[0] * pred_rel[0] - a_lhs[1] * pred_rel[1]) @ to_score[0].transpose(0, 1) +
                            (a_lhs[0] * pred_rel[1] + a_lhs[1] * pred_rel[0]) @ to_score[1].transpose(0, 1) +
                            (a_lhs_ent * pred_rel_rel) @ to_score_ent.transpose(0, 1)
                    ), (
                        torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2),
                        torch.sqrt(pred_rel[0] ** 2 + pred_rel[1] ** 2),
                        torch.sqrt(a_lhs[0] ** 2 + a_lhs[1] ** 2),
                        torch.sqrt(a_lhs_ent ** 2 + pred_rel_rel ** 2)
                    )
                                     
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            return self.r_embeddings[0].weight.data[
                   chunk_begin:chunk_begin + chunk_size
                   ],img_embeddings
        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            return embedding[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)
            
    def get_relations(self):
        return self.r_embeddings[1].weight.transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(queries[:, 1])]) * self.r_embeddings[0](queries[:, 0]) + self.alpha[(queries[:, 1])] * img_embeddings[
                (queries[:, 0])]
            rel = self.r_embeddings[1](queries[:, 1])

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)
        else:
            rel = self.r_embeddings[1](queries[:, 1])
            rel_rel = self.rel_embeddings(queries[:, 1])
            
            lhs = torch.rand((len(queries), 2*self.rank)).to(queries.device)
            lhs_ent = torch.rand((len(queries), 2*self.rank)).to(queries.device)
            for i in range(len(queries)):
                mode = queries[i, -1].item()
                if mode == 0:   # （T， T）
                    lhs_ent[i] = self.ent_embeddings(queries[i, 0])
                    lhs[i] = self.r_embeddings[0](queries[i, 0])
                elif mode == 1: # (I, T)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                else:           # (I, I)
                    lhs_ent[i] = (1 - self.alpha) * self.ent_embeddings(queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]
                    lhs[i] = (1 - self.alpha) * self.r_embeddings[0](queries[i, 0]) + self.alpha * img_embeddings[queries[i, 0]]


            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1) + lhs_ent * rel_rel

