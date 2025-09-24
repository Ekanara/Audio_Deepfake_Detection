import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler
import json
from collections import defaultdict
import random
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoProcessor, CLIPModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
import torch

class BaselineDataset():
    def __init__(self, json_file, model=None ,transformation=None, device='cuda'):
        with open(file=json_file, mode='r') as f:
            self.data = json.load(f)
        self.model= model
        if self.model:
            self.model = self.model.to(device) 
        self.label = {label: idx for idx, label in enumerate(sorted(set(d['label'] for d in self.data)))}

        self.data.sort(key=lambda x: self.label[x['label']])        
        self.transformation = transformation
        self.device = device

    def CLIP_extract(self, input):
        with torch.no_grad():
            #input = self.proccessor(images=input, return_tensors='pt')
            #output = self.model.get_image_features(**input)
            output = self.model.forward_features(input.unsqueeze(0))
            return output
    
    def load_image(self,image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def preprocess(self):
        images_list = []
        label_list = []
        for i, item in enumerate(self.data):
            image_path = item['image_path']
            label = self.label[item['label']]

            image = self.load_image(image_path)
            if self.transformation:
                image = self.transformation(image)
            
            image = image.to(self.device)
            image = self.CLIP_extract(image)
            
            images_list.append(image.squeeze(0).to('cpu'))
            label_list.append(label)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(self.data)} images")
        
        images_tensor = torch.stack(images_list)
        labels_tensor = torch.tensor(label_list, dtype=torch.long)
        dataset = {image: images_tensor,
                  label: labels_tensor}
        
        print(f"Dataset created with {len(images_list)} images")
        print(f"Image features shape: {images_tensor.shape}")
        print(f"Labels shape: {labels_tensor.shape}")
        return dataset
    
    def save_dataset(self, output_path):
        """Process and save the dataset"""
        dataset = self.preprocess()
        hf_dataset = Dataset.from_dict(dataset)
        hf_dataset.set_format(type='torch')
        hf_dataset.save_to_disk(output_path)
        print(f"Dataset saved to {output_path}")
        return dataset
    
if __name__ == '__main__':
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=1.0)
        ], p=0.5),  # flip prob

        transforms.RandomApply([
            transforms.RandomRotation(degrees=10)
        ], p=0.5),  # rotate prob, limit [-10, 10]

        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1)
        ], p=0.5),  # brightness prob

        transforms.RandomApply([
            transforms.ColorJitter(contrast=0.1)
        ], p=0.5),  # contrast prob

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    json_file = 'data/train/image_labels.json'
    model = timm.create_model('resnet50_clip.openai', pretrained=True)
    baseline_dataset = BaselineDataset(json_file=json_file, model = model, transformation=transformation)
    processed_data = baseline_dataset.preprocess()

    # Or save directly:
    baseline_dataset.save_dataset('data/dataset_stage1')
    # print('Get process_data')
    # processed_data = list(baseline_dataset)

    # print("Saving data")
    # torch.save(processed_data,'data_stage1.pt')

    # print('Load process data')
    # processed_data = torch.load('data_stage1.pt')

    # print('Done loading')
    # samples_per_label = {'efs': 2, 'fr': 2, 'fs': 2, 'real': 2}  # batch size = 8
    # sampler = BalancedGeneratorSampler(processed_data, samples_per_label)

    # dataloader = DataLoader(processed_data, batch_sampler=sampler)

    # for batch in dataloader:
    #         image = batch['image']
    #         print(image.shape)