# encode images as torch tensor
from transformers import CLIPProcessor, ViltProcessor
from tqdm import tqdm
from PIL import Image
import os
import torch
import random
random.seed(1)

base_path = 'dataset/MARS/images'
img_path = os.listdir(base_path)
processor_name = 'clip'  # clip or vilt
print(len(img_path))

if processor_name == 'clip':
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    entity2visual = []
    for entity in tqdm(img_path, total=len(img_path)):
        path = os.path.join(base_path, entity)
        sub_files = os.listdir(path)
        sub_file = random.sample(sub_files, k=1)[0]
        image = Image.open(os.path.join(path, sub_file)).convert('RGB')
        piexel_values = processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
        entity2visual.append(piexel_values)

    torch.save(torch.stack(entity2visual, dim=0), 'dataset/MARS/entity_image_features.CLIP-VIT-16-32.pth')
        
elif processor_name == 'vilt':
    entity2visual = []

    processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')

    for entity in tqdm(img_path, total=len(img_path)):
        path = os.path.join(base_path, entity)
        sub_files = os.listdir(path)
        file = random.sample(sub_files, k=1)[0]
        image = Image.open(os.path.join(path, file)).convert('RGB').resize((384, 384))
        pixel_values = processor(images=image, text="test", return_tensors='pt')['pixel_values']
        entity2visual.append(pixel_values)

    entity2visual = torch.cat(entity2visual, dim=0)
    torch.save(entity2visual, 'dataset/MARS/entity_image_features_vilt.pth')
