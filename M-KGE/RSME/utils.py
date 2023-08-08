import numpy as np
import os
import random
import pickle
import math


def get_img_vec_array(proportion,img_vec_path='fb15k_vit.pickle',eutput_file='img_vec_id_fb15k_{}_vit.pickle',dim=1000):
    img_vec=pickle.load(open(img_vec_path,'rb'))
    img_vec={k.split('/')[-2]:v for k,v in img_vec.items()}
    f=open('src_data/Analogy/wiki_tuple_ids', 'r')
    Lines=f.readlines()
    entities = set()
    for line in Lines:
        head, rel, tail = line.split('\t')
        entities.add(head)
        entities.add(tail.replace('\n', ''))

    id2ent={}
    img_vec_array=[]
    for id, ent in enumerate(entities):
        id2ent[id]=ent
        if ent in img_vec.keys():
            print(id, ent)
            img_vec_array.append(img_vec[ent])
        else:
            img_vec_array.append([0 for i in range(dim)])
    img_vec_by_id = np.array(img_vec_array)
    out=open(eutput_file,'wb')
    pickle.dump(img_vec_by_id,out)
    out.close()


def get_img_vec_array_forget(proportion,remember_proportion,rank_file='fb15k_vit_rank.txt',eutput_file='rel_MPR_PD_vit_{}_mrp{}.pickle'):
    with open(rank_file,'r') as f:
        Ranks=f.readlines()
        rel_rank={}
        for r in Ranks:
            try:
                rel,mrp=r.strip().split('\t')
            except Exception as e:
                print(e)
                print(r)
                continue
            rel_rank[rel[10:]]=float(mrp[12:])

    with open('../../MarT/dataset/MarKG/relation2text.txt', 'r') as f:
        Lines=f.readlines()

    rel_id_pd=[]
    for l in Lines:
        rel,_=l.strip().split('\t')
        try:
            if rel_rank[rel]<remember_proportion/100.0:
                rel_id_pd.append([1])
            else:
                rel_id_pd.append([0])
        except Exception as e:
            print(e)
            rel_id_pd.append([0])
            continue

    rel_id_pd=np.array(rel_id_pd)

    with open(eutput_file.format(remember_proportion),'wb') as out:
        pickle.dump(rel_id_pd,out)


def get_img_vec_sig_alpha(proportion,rank_file='fb15k_vit_rank.txt',eutput_file='rel_MPR_SIG_vit_{}.pickle'):
    with open(rank_file,'r') as f:
        Ranks=f.readlines()
        rel_rank={}
        for r in Ranks:
            try:
                rel,mrp=r.strip().split('\t')
            except Exception as e:
                print(e)
                print(r)
                continue
            rel_rank[rel[10:]]=float(mrp[12:])

    with open('../../MarT/dataset/MarKG/relation2text.txt', 'r') as f:
        Lines=f.readlines()

    rel_sig_alpha=[]
    for l in Lines:
        rel,_=l.strip().split('\t')
        try:
            rel_sig_alpha.append([1/(1+math.exp(rel_rank[rel]))])
        except Exception as e:
            print(e)
            rel_sig_alpha.append([1 / (1 + math.exp(1))])
            continue

    rel_id_pd=np.array(rel_sig_alpha)

    with open(eutput_file,'wb') as out:
        pickle.dump(rel_id_pd,out)

def sample(proportion,data_path='./src_data/FB15K'):
    with open(data_path+'/train') as f:
        Ls=f.readlines()
        L = [random.randint(0, len(Ls)-1) for _ in range(round(len(Ls)*proportion))]
        Lf=[Ls[l] for l in L]

    if not os.path.exists(data_path+'_{}/'.format(round(proportion*100))):
        os.mkdir(data_path+'_{}/'.format(round(proportion*100)))
    Ent=set()

    with open(data_path+'_{}/train'.format(round(100*proportion)),'w') as f:
        for l in Lf:
            h,r,t=l.strip().split()
            Ent.add(h)
            Ent.add(r)
            Ent.add(t)
            f.write(l)
            f.flush()

    with open(data_path+'/valid','r') as f:
        Ls = f.readlines()

    with open(data_path+'_{}/valid'.format(round(100*proportion)),'w') as f:
        for l in Ls:
            h,r,t=l.strip().split()
            if h in Ent and r in Ent and t in Ent:
                f.write(l)
                f.flush()
            else:
                print(l.strip()+' pass')

    with open(data_path+'/test','r') as f:
        Ls = f.readlines()

    with open(data_path+'_{}/test'.format(round(proportion*100)),'w') as f:
        for l in Ls:
            h, r, t = l.strip().split()
            if h in Ent and r in Ent and t in Ent:
                f.write(l)
                f.flush()
            else:
                print(l.strip()+' pass')
                
def split_mkg_data(root_path):
    for f in ['train', 'valid', 'test']:
        path = open(f'{root_path}/{f}.pickle', 'rb')
        data = pickle.load(path)
        data = np.append(data, np.zeros((len(data), 1)), axis=-1)
        for i in range(len(data)):
            rnd = random.random()
            if rnd <= 0.4:
                data[i][-1] = 0
            elif rnd > 0.4 and rnd < 0.7:
                data[i][-1] = 1
            else:
                data[i][-1] = 2
        with open(f'{root_path}/{f}.pickle', 'wb') as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    get_img_vec_array(0, img_vec_path='data/analogy/analogy_vit_best_img_vec.pickle', eutput_file='data/analogy/img_vec_id_analogy_vit.pickle')
    get_img_vec_sig_alpha(20, 'data/analogy/analogy_vit_rank.txt', 'data/analogy/rel_MPR_SIG_vit.pickle')
    get_img_vec_array_forget(30, 100, 'data/analogy/analogy_vit_rank.txt', 'data/analogy/rel_MPR_PD_vit_mrp{}.pickle')
    split_mkg_data('data/analogy')



