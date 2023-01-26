# -*- coding: UTF-8 -*-
'''
Image encoder.  Get visual embeddings of images.
'''
import os
import imagehash
from PIL import Image
import pickle
from tqdm import tqdm
class FilterGate():
    def __init__(self,base_path,hash_size):
        self.base_path=base_path
        self.hash_size=hash_size
        self.best_imgs={}

    def phash_sim(self,img1,img2,hash_size=None):
        if not hash_size:
            hash_size=self.hash_size
        img1_hash = imagehash.phash(Image.open(img1), hash_size=hash_size)
        img2_hash = imagehash.phash(Image.open(img2), hash_size=hash_size)

        return 1 - (img1_hash - img2_hash) / len(img1_hash) ** 2

    def filter(self):
        self.best_imgs={}
        ents = os.listdir(self.base_path)
        pbar = tqdm(total=len(ents))
        while len(ents)>0:
            ent=ents.pop()
            imgs=os.listdir(self.base_path + ent + '/')
            n_img=len(imgs)
            if n_img == 0:
                pbar.update(1)
                continue
            sim_matrix=[[0]*n_img for i in range(n_img)]
            for i in range(n_img):
                for j in range(i+1,n_img):
                    sim=self.phash_sim(self.base_path + ent + '/'+imgs[i], self.base_path + ent + '/'+imgs[j])
                    sim_matrix[i][j]=sim
                    sim_matrix[j][i] =sim
            max_index=0
            max_sim=sum(sim_matrix[0])
            for i in range(1,n_img):
                if sum(sim_matrix[i])>max_sim:
                    max_index=i
                    max_sim=sum(sim_matrix[i])
            self.best_imgs[ent]=self.base_path + ent + '/'+imgs[max_index]
            pbar.update(1)
        pbar.close()
        return self.best_imgs

    def save_best_imgs(self,output_file,n=1):
        with open(output_file, 'wb') as out:
            pickle.dump(self.best_imgs, out)




if __name__ == '__main__':
    f=FilterGate('../MarT/dataset/MARS/images/', hash_size=16)
    f.filter()
    f.save_best_imgs('analogy_best_img.pickle')











