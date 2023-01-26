import torchvision.models as models
from torchvision import datasets, transforms
import torch
import collections
from PIL import Image
import glob
import pickle
from tqdm import tqdm
import os
vgg16 = models.vgg16(pretrained=True)


vgg16.features = torch.nn.Sequential(collections.OrderedDict(zip(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'], vgg16.features)))
vgg16.classifier = torch.nn.Sequential(collections.OrderedDict(zip(['fc6', 'relu6', 'drop6', 'fc7'], vgg16.classifier)))

print(vgg16)

# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transformation_model = transforms.Compose([
    transforms.Resize((224,224), interpolation=0),
    transforms.ToTensor(),
    normalize
])

all_input_img = []
entity_ids = []
batch_size = 0 
emb_dim = 1024
batch_dim = 128
with open("../MarT/dataset/MarKG/entity2text.txt", "r") as enidf:
    for line in tqdm(enidf.readlines()):
        entity = line.split('\t')[0]
        input_img = []
        for filename in glob.glob("../MarT/dataset/MARS/images/"+entity+"/*.*"):
            try:
                im=Image.open(filename)
                im = transformation_model(im)
                input_img.append(im)
            except:
                pass
                
        if len(input_img) > 0 and batch_size + len(input_img) <= batch_dim:
            input_img = torch.stack(input_img, dim=0) 
            all_input_img.append(input_img)
            batch_size += len(input_img)
            entity_ids.append(entity)
        elif len(input_img) > 0 and batch_size + len(input_img) > batch_dim:
            lengths = [len(item) for item in all_input_img]
            vgg_input = torch.cat(all_input_img, dim=0) 
            result = vgg16(vgg_input)
            results_split = torch.split(result, lengths)
            for index, item in enumerate(results_split):
                embed = item.mean(0)
                if not os.path.exists("data/analogy/" + entity_ids[index]):
                    os.makedirs("data/analogy/" + entity_ids[index])
                with open("data/analogy/" + entity_ids[index] + "/avg_embedding.pkl", "wb+") as f:
                    pickle.dump(embed, f)
                
            # reinitialize
            all_input_img = []
            batch_size = 0 
            entity_ids = []
            # add the remanent into it
            input_img = torch.stack(input_img, dim=0) 
            all_input_img.append(input_img)
            batch_size += len(input_img)
            entity_ids.append(entity)
            
