import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
import json
from collections import defaultdict
import random
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, CLIPModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
import itertools
import torch
from src.base_dataset import BaselineDataset

class ENVITDataset(Dataset):
    def __init__(self, json_file, transformation=None, device='cuda'):
        with open(file=json_file, mode='r') as f:
            self.data = json.load(f)
        # Create label mapping (efs=1, fr=2, fs=3, real=4)
        self.label = {label: idx for idx, label in enumerate(sorted(set(d['label'] for d in self.data)))}
        # self.model = CLIPModel.from_pretrained("timm/resnet50_clip.openai")
        # self.proccessor = AutoProcessor.from_pretrained("timm/resnet50_clip.openai")
        #self.model = timm.create_model('resnet50_clip.openai', pretrained=True)
        # Sort dataset by label
        self.data.sort(key=lambda x: self.label[x['label']])        
        self.transformation = transformation
        self.device = device
    
    def __len__(self):
        return len(self.data) 

    def __getitem__(self,idx):
        seed = 42 + idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        item = self.data[idx]
        audio_path = item['audio']
        label = self.label[item['label']]

        audio = np.load(audio_path)
        audio = torch.from_numpy(audio)
        #print(audio.shape)
        audio = audio.squeeze(0).to(torch.bfloat16)
        #print(audio.shape)
        # if audio.shape[0] == 3 and audio.shape[-1] != 3:  # CHW
        #     audio = audio.permute(2, 1, 0)
        #     print(audio.shape)
            # quit()
        #print(audio.shape)
        #print(audio.shape)
        #quit()
        # print(audio.shape)
        # print("Min:", audio.min())
        # print("Max:", audio.max())

        # quit()
        # if self.transformation:
        #     image = self.transformation(image)  
        
        if self.transformation:
            # Albumentations expects dict, torchvision does not
            if hasattr(self.transformation, "__call__") and "albumentations" in str(type(self.transformation)).lower():
                image = np.array(image)
                transformed = self.transformation(image=image) 
                image = transformed["image"]

            else: # This for normal torchvision
                audio = self.transformation(audio)
        
        print(audio.shape)
        quit()
            # # Albumentations returns dict
            # if isinstance(transformed, dict):
            # else:
            #     image = transformed
        
        # image = image.to(self.device)
        # label = label.to(self.device)
        return {"audio": audio, 
                "label": label,
                "path": audio_path}

class BalancedGeneratorSampler(Sampler):
    def __init__(self, dataset, samples_per_label):
        """
        Balanced sampler that ensures each batch contains a fixed number of samples
        per label, and cycles through different generators (subfolders) for each label.

        Args:
            dataset: Dataset object with a .data list of dicts 
                     containing 'label' and 'image'.
            samples_per_label: dict mapping label_name -> number of samples per batch.
                               Example: {'efs': 2, 'fr': 2, 'fs': 2, 'real': 2}
        """
        self.dataset = dataset
        self.samples_per_label = samples_per_label  

        # label -> generator -> list of dataset indices
        self.label_to_gen_indices = defaultdict(lambda: defaultdict(list))
        for idx, item in enumerate(dataset.data):
            label_name = item['label']  # string label from JSON
            gen_name = os.path.basename(os.path.dirname(item['audio']))  # subfolder name
            self.label_to_gen_indices[label_name][gen_name].append(idx)

        # Shuffle the image indices for each generator
        for label in self.label_to_gen_indices:
            for gen in self.label_to_gen_indices[label]:
                random.shuffle(self.label_to_gen_indices[label][gen])

        self.labels = list(samples_per_label.keys())

    def __iter__(self):
        # Track pointers for each generator's image list
        gen_pointers = {
            label: {gen: 0 for gen in self.label_to_gen_indices[label]}
            for label in self.labels
        }

        while True:
            batch = []
            for label in self.labels:
                gens = list(self.label_to_gen_indices[label].keys())
                random.shuffle(gens)  # shuffle generator order

                picked = 0
                while picked < self.samples_per_label[label]:
                    for gen in gens:
                        if picked >= self.samples_per_label[label]:
                            break
                        ptr = gen_pointers[label][gen]
                        audio = self.label_to_gen_indices[label][gen]
                        if ptr >= len(audio):
                            return  # stop iteration if one generator runs out
                        batch.append(audio[ptr])
                        gen_pointers[label][gen] += 1
                        picked += 1
            yield batch

    def __len__(self):
        return min(
            sum(len(audio) for audio in self.label_to_gen_indices[label].values())
            // self.samples_per_label[label]
            for label in self.labels
        )
    
class BalancedGeneratorSampler_Stage2(Sampler):
    def __init__(self, dataset, samples_per_generator=1):
        """
        Balanced sampler for real vs fake.
        Each fake generator contributes `samples_per_generator` samples,
        and real contributes the total sum of those.
        
        Args:
            dataset: Dataset with dataset.data = [{'label': 'real'/'fake', 'image': ...}]
            samples_per_generator: int, number of samples to take from each generator per batch.
        """
        self.dataset = dataset
        self.samples_per_generator = samples_per_generator

        # Collect dataset indices
        self.fake_gen_to_indices = defaultdict(list)  # generator -> indices
        self.real_indices = []

        for idx, item in enumerate(dataset.data):
            label = item['label']
            if label == "real":
                self.real_indices.append(idx)
            else:  # fake
                gen_name = os.path.basename(os.path.dirname(item['image']))
                self.fake_gen_to_indices[gen_name].append(idx)

        # Shuffle
        for gen in self.fake_gen_to_indices:
            random.shuffle(self.fake_gen_to_indices[gen])
        random.shuffle(self.real_indices)

        # Pointers
        self.fake_pointers = {gen: 0 for gen in self.fake_gen_to_indices}
        self.real_pointer = 0
        self.generators = list(self.fake_gen_to_indices.keys())

    def __iter__(self):
        # Reset pointers each epoch
        self.fake_pointers = {gen: 0 for gen in self.fake_gen_to_indices}
        self.real_pointer = 0

        while True:
            batch = []
            total_fake = 0

            # Add fake samples
            for gen in self.generators:
                for _ in range(self.samples_per_generator):
                    ptr = self.fake_pointers[gen]
                    audio = self.fake_gen_to_indices[gen]
                    if ptr >= len(audio):
                        return
                    batch.append(audio[ptr])
                    self.fake_pointers[gen] += 1
                total_fake += self.samples_per_generator

            # Add real samples
            if self.real_pointer + total_fake > len(self.real_indices):
                return
            batch.extend(self.real_indices[self.real_pointer:self.real_pointer + total_fake])
            self.real_pointer += total_fake

            yield batch
        

    def __len__(self):
        fake_batches = min(
            len(self.fake_gen_to_indices[gen]) // self.samples_per_generator
            for gen in self.generators
        )
        real_batches = len(self.real_indices) // (len(self.generators) * self.samples_per_generator)
        return min(fake_batches, real_batches)
    
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, shuffle=False):
        """
        labels: list/array/tensor of 0 (fake) or 1 (real) for each sample
        """
        self.labels = labels
        self.shuffle = shuffle

        self.real_indices = [i for i, l in enumerate(labels) if l == 1]
        self.fake_indices = [i for i, l in enumerate(labels) if l == 0]

        self.max_len = max(len(self.real_indices), len(self.fake_indices))

    def __iter__(self):
        real = self.real_indices
        fake = self.fake_indices

        if self.shuffle:
            real = random.sample(real, len(real))
            fake = random.sample(fake, len(fake))

        # yield pairs
        for i in range(self.max_len):
            yield real[i % len(real)]
            yield fake[i % len(fake)]

    def __len__(self):
        return self.max_len * 2

def save_dataset_to_parquet(examples, parquet_path="dataset.parquet"):
    images = []
    labels = []

    for i in range(len(examples)):
        sample = baseline_dataset[i]
        images.append(sample['image'])
        labels.append(sample['label'])

    images = torch.stack(images).to('cpu')
    labels = torch.stack(labels).to('cpu')
    # Build HF Dataset directly from tensors
    hf_dataset = Dataset.from_dict({
        "image": images,
        "label": labels
    })

    # Ensure output stays as torch tensors when loading
    hf_dataset.set_format(type="torch", columns=["image", "label"])

    # Save to parquet
    hf_dataset.to_parquet(parquet_path)
    print(f"Dataset saved to {parquet_path}")

def load_dataset_from_disk(file_path="processed_dataset.parquet"):
    dataset = Dataset.from_parquet(file_path)
    return dataset


if __name__ == '__main__':
    transformation = transforms.Compose([
        # transforms.Resize((224, 224)),

        # transforms.RandomApply([
        #     transforms.RandomHorizontalFlip(p=1.0)
        # ], p=0.5),  # flip prob

        # transforms.RandomApply([
        #     transforms.RandomRotation(degrees=10)
        # ], p=0.5),  # rotate prob, limit [-10, 10]

        # transforms.RandomApply([
        #     transforms.ColorJitter(brightness=0.1)
        # ], p=0.5),  # brightness prob

        # transforms.RandomApply([
        #     transforms.ColorJitter(contrast=0.1)
        # ], p=0.5),  # contrast prob

        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    json_file = 'data/audio_labels.json'
    model = timm.create_model('resnet50_clip.openai', pretrained=True)
    baseline_dataset = BaselineDataset(json_file=json_file, transformation=transformation)
    
    print('Get process_data')
    processed_data = list(baseline_dataset)

    print("Saving data")
    torch.save(processed_data,'data_stage1.pt')

    print('Load process data')
    processed_data = torch.load('data_stage1.pt')

    print('Done loading')
    samples_per_label = {'efs': 2, 'fr': 2, 'fs': 2, 'real': 2}  # batch size = 8
    sampler = BalancedGeneratorSampler(processed_data, samples_per_label)

    dataloader = DataLoader(processed_data, batch_sampler=sampler)

    for batch in dataloader:
            image = batch['image']
            print(image.shape)

    # dataset = dataset.map(dataset, batched=True, batch_size=1)
    # dataset.set_format(type='torch')

    # samples_per_label = {'efs': 2, 'fr': 2, 'fs': 2, 'real': 2}  # batch size = 8
    # sampler = BalancedGeneratorSampler(dataset, samples_per_label)

    # dataloader = DataLoader(dataset, batch_sampler=sampler)
    
    # for batch in dataloader:
    #     image = batch['image']
    #     print(image.shape)
    #     break
    # # for images, labels in dataset:
    # #     print(images.shape, labels.shape)
    # #     quit()
    
    # #save_dataloader_to_disk(dataloader=dataloader, output_file='data/dataset_stage1.pt')