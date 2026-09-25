import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
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
import torch.nn.functional as F
import librosa
import soundfile as sf

class BaselineDataset(Dataset):
    def __init__(self, json_file, transformation=None, device='cuda', args=None):
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
        self.args = args
    
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
        #label = torch.nn.functional.one_hot(label, num_classes=self.args.num_label).float()

        audio = np.load(audio_path)
        audio = torch.from_numpy(audio)
        audio = audio.squeeze(0).to(torch.bfloat16)

        
        if self.args.get_first_dim:
            audio = audio[0:1, :, :]
            audio = torch.stack([audio, audio, audio], dim=1).squeeze(0)
            
        if self.transformation:
            if hasattr(self.transformation, "__call__") and "albumentations" in str(type(self.transformation)).lower():
                audio = np.array(audio)
                transformed = self.transformation(image=audio) 
                audio = transformed["audio"]

            else: # This for normal torchvision
                audio = self.transformation(audio)

        if self.args.mode == 'inception':
            audio = F.interpolate(audio.unsqueeze(0), scale_factor=3, mode='bilinear', align_corners=False).squeeze(0)

        return {"audio": audio, 
                "label": label,
                "path": audio_path}
    
class BeatsDataset(Dataset):
    def __init__(self, json_file, transformation=None, device='cuda', args=None):
        with open(file=json_file, mode='r') as f:
            self.data = json.load(f)
        
        # Create label mapping
        self.label = {label: idx for idx, label in enumerate(sorted(set(d['label'] for d in self.data)))}
        
        # Sort dataset by label
        self.data.sort(key=lambda x: self.label[x['label']])        
        self.transformation = transformation
        self.device = device
        self.args = args
        
        # Audio processing parameters
        self.fs = getattr(args, 'sample_rate', 16000)  # Default sample rate
        self.set_dur = getattr(args, 'segment_duration', 4)  # Default 4 seconds
        self.return_segments = getattr(args, 'return_segments', False)  # Return all segments or just first one
    
    def __len__(self):
        return len(self.data) 

    def get_file_content(self, file_path):
        """
        Read audio file and process it into overlapping segments
        """
        # Read audio file (supports .wav and .flac)
        wav, org_fs = sf.read(file_path)

        # Convert to mono if stereo
        if wav.ndim > 1:
            wav = wav[:, 0]
        
        # Resample if necessary
        if org_fs != self.fs:
            wav = librosa.core.resample(wav, orig_sr=org_fs, target_sr=self.fs)

        wav = np.asarray(wav)
        nTime = np.shape(wav)[0]

        # Required length for segment
        nT = self.set_dur * self.fs
        
        # Repeat audio if too short
        if nTime < nT:
            while True:
                wav = np.concatenate((wav, wav), axis=-1)
                nTime = np.shape(wav)[0]
                if nTime > nT:
                    break

        # Split into overlapping segments
        nTime = np.shape(wav)[0]
        split_num = 2 + np.floor((nTime - nT) * 2 / nT)  # overlapping
        
        mul_seg_sample = None
        for m in range(int(split_num)):
            if m == split_num - 1:
                tStop = nTime
                tStart = nTime - nT
            else:
                tStart = int(m * nT / 2)
                tStop = tStart + nT

            if m == 0:
                mul_seg_sample = wav[tStart:tStop]
                mul_seg_sample = np.reshape(mul_seg_sample, (1, -1))
            else:
                mul_seg_sample = np.concatenate((mul_seg_sample, wav[tStart:tStop].reshape(1, -1)), 0)

        return mul_seg_sample  # Shape: (num_segments, segment_length)

    def __getitem__(self, idx):
        seed = 42 + idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        item = self.data[idx]
        audio_path = item['audio']
        label = self.label[item['label']]

        # Load and process audio file
        audio_segments = self.get_file_content(audio_path)
        
        # For BEATs model, typically we want just the first segment as 1D tensor
        # If you need all segments, set args.return_segments = True
        if self.return_segments:
            audio = torch.from_numpy(audio_segments)
            audio = audio.to(torch.float32)
        else:
            # Return only first segment as 1D tensor (shape: segment_length)
            audio = torch.from_numpy(audio_segments[0])
            audio = audio.to(torch.float32)
        
        # Apply transformations if provided
        if self.transformation:
            if hasattr(self.transformation, "__call__") and "albumentations" in str(type(self.transformation)).lower():
                audio = np.array(audio)
                transformed = self.transformation(image=audio) 
                audio = transformed["audio"]
            else:  # This for normal torchvision
                audio = self.transformation(audio)

        return {
            "audio": audio, 
            "label": label,
            "path": audio_path
        }
    
class RatioSampler(Sampler):
    def __init__(self, dataset, batch_size=36):
        if batch_size % 9 != 0:
            raise ValueError(f"batch_size must be a multiple of 9, got {batch_size}")
        self.group_size = 9  # 1 real + 8 fake per group
        self.groups_per_batch = batch_size // self.group_size

        self.real_idx = [i for i, x in enumerate(dataset.data) if x['label'] == 'real']
        self.fake_idx = [i for i, x in enumerate(dataset.data) if x['label'] == 'fake']

    def __iter__(self):
        random.shuffle(self.real_idx)
        random.shuffle(self.fake_idx)

        num_groups = min(len(self.real_idx), len(self.fake_idx) // 8)
        total_batches = num_groups // self.groups_per_batch

        for b in range(total_batches):
            batch_indices = []
            for g in range(self.groups_per_batch):
                group_id = b * self.groups_per_batch + g
                real_sample = [self.real_idx[group_id]]
                fake_samples = self.fake_idx[group_id * 8 : (group_id + 1) * 8]
                batch_indices.extend(real_sample + fake_samples)
            yield batch_indices

    def __len__(self):
        num_groups = min(len(self.real_idx), len(self.fake_idx) // 8)
        return num_groups // self.groups_per_batch
    
# class RatioSampler(Sampler):
#     def __init__(self, dataset, batch_size=32):
#         if batch_size % 8 != 0:
#             raise ValueError(f"batch_size must be a multiple of 8, got {batch_size}")
#         self.group_size = 8  # 4 real + 4 fake per group
#         self.groups_per_batch = batch_size // self.group_size

#         # indices
#         self.real_idx = [i for i, x in enumerate(dataset.data) if x['label'] == 'real']
#         self.fake_idx = [i for i, x in enumerate(dataset.data) if x['label'] == 'fake']

#     def __iter__(self):
#         random.shuffle(self.real_idx)
#         random.shuffle(self.fake_idx)

#         # number of groups = min(number of unique reals, number of fake groups of 4)
#         num_groups = min(len(self.real_idx), len(self.fake_idx) // 4)
#         total_batches = num_groups // self.groups_per_batch

#         for b in range(total_batches):
#             batch_indices = []
#             for g in range(self.groups_per_batch):
#                 group_id = b * self.groups_per_batch + g

#                 # pick 1 real for this group
#                 real_sample_idx = self.real_idx[group_id % len(self.real_idx)]

#                 # 4 fakes for this group
#                 fake_start = (group_id * 4) % len(self.fake_idx)
#                 fake_samples = self.fake_idx[fake_start:fake_start + 4]

#                 # group indices: just 1 real (the collate function will duplicate 3 + 1 gen_real)
#                 batch_indices.extend([real_sample_idx] + fake_samples)

#             yield batch_indices

#     def __len__(self):
#         num_groups = min(len(self.real_idx), len(self.fake_idx) // 4)
#         return (num_groups // self.groups_per_batch) * self.groups_per_batch
    

# def mixup_collate_fn(batch, lam=0.8):
#     audios = torch.stack([b["audio"] for b in batch])
#     labels = torch.tensor([b["label"] for b in batch])

#     real_audios = audios[labels == 1]
#     fake_audios = audios[labels == 0]

#     batch_size = len(audios)
#     groups_per_batch = batch_size // 8
#     batch_audio_list = []
#     batch_label_list = []

#     for g in range(groups_per_batch):
#         # pick 1 real randomly
#         real_idx = g % len(real_audios)
#         real_audio = real_audios[real_idx]

#         # 3 duplicates + 1 gen_real
#         real_copies = torch.stack([real_audio]*3)
#         gen_real = lam * real_audio + (1 - lam) * fake_audios[g % len(fake_audios)]
#         real_group = torch.cat([real_copies, gen_real.unsqueeze(0)], dim=0)
#         batch_audio_list.append(real_group)
#         batch_label_list.append(torch.tensor([1]*4))

#         # 4 fakes
#         fake_start = (g*4) % len(fake_audios)
#         fake_group = fake_audios[fake_start:fake_start+4]
#         batch_audio_list.append(fake_group)
#         batch_label_list.append(torch.tensor([0]*4))

#     mixed_audios = torch.cat(batch_audio_list, dim=0)
#     mixed_labels = torch.cat(batch_label_list, dim=0)

#     return {"audio": mixed_audios, "label": mixed_labels}


def mixup_collate_fn(batch, lam_range=(0.7, 0.95)):
    audios = torch.stack([b["audio"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    wav_names = [os.path.basename(b["path"]) for b in batch]  # ← use existing "path" key

    grouped = {}
    for audio, label, wav_name in zip(audios, labels, wav_names):
        if wav_name not in grouped:
            grouped[wav_name] = {"real": None, "fakes": []}
        if label == 1:
            grouped[wav_name]["real"] = audio
        else:
            grouped[wav_name]["fakes"].append(audio)

    batch_audio_list = []
    batch_label_list = []

    for wav_name, group in grouped.items():
        real_audio = group["real"]
        fakes = group["fakes"]

        if real_audio is None or len(fakes) == 0:
            continue

        lam = random.uniform(*lam_range)

        gen_reals = torch.stack([
            lam * real_audio + (1 - lam) * fake
            for fake in fakes
        ])

        true_fakes = torch.stack(fakes)

        group_audios = torch.cat([
            real_audio.unsqueeze(0),  # 1 real         (label=1)
            gen_reals,                # 4 gen_reals     (label=1)
            true_fakes,               # 4 true fakes    (label=0)
        ], dim=0)

        num_fakes = len(fakes)
        group_labels = torch.tensor([1] * (1 + num_fakes) + [0] * num_fakes)

        batch_audio_list.append(group_audios)
        batch_label_list.append(group_labels)

    if not batch_audio_list:
        return {"audio": audios, "label": labels}

    return {
        "audio": torch.cat(batch_audio_list, dim=0),
        "label": torch.cat(batch_label_list, dim=0),
    }

def val_collate_fn(batch):
    audios = torch.stack([b["audio"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    return {"audio": audios, "label": labels}

class BalancedGeneratorSampler(BatchSampler):
    """
    Balanced sampler that yields fixed-composition batches.

    Each batch contains `samples_per_label[label]` indices for every label.
    Epoch length is determined by the LARGEST label (in batches); smaller
    labels cycle — when a generator's pool is exhausted it gets reshuffled
    and rewound to start. This way the under-represented label (e.g. real)
    is duplicated across the epoch instead of cutting the epoch short.

    Pass cycle_short_labels=False to recover the old behaviour where the
    epoch length was the MIN across labels.
    """
    def __init__(self, dataset, samples_per_label, cycle_short_labels=True):
        """
        Args:
            dataset: Dataset object
            samples_per_label: dict mapping label_name -> samples per batch
                               Example: {'fake': 80, 'real': 80}
            cycle_short_labels: if True (default), exhausted labels cycle and
                                epoch length = max over labels. If False, the
                                epoch ends with the smallest label.
        """
        self.dataset = dataset
        self.samples_per_label = samples_per_label
        self.batch_size = sum(samples_per_label.values())
        self.cycle_short_labels = cycle_short_labels

        # Build label -> generator -> indices mapping
        self.label_to_gen_indices = defaultdict(lambda: defaultdict(list))
        for idx, item in enumerate(dataset.data):
            label_name = item['label']
            gen_name = os.path.basename(os.path.dirname(item['audio']))
            self.label_to_gen_indices[label_name][gen_name].append(idx)

        self.labels = list(samples_per_label.keys())
        self._calculate_length()

    def _calculate_length(self):
        """Number of batches per epoch: max across labels (cycle mode) or min."""
        per_label_batches = []
        for label in self.labels:
            total_for_label = sum(
                len(indices) for indices in self.label_to_gen_indices[label].values()
            )
            per_label_batches.append(total_for_label // self.samples_per_label[label])

        self.num_batches = int(max(per_label_batches) if self.cycle_short_labels
                                else min(per_label_batches))

    def __iter__(self):
        # Prepare shuffled indices for each generator
        gen_indices_copy = defaultdict(lambda: defaultdict(list))
        for label in self.label_to_gen_indices:
            for gen in self.label_to_gen_indices[label]:
                gen_indices_copy[label][gen] = self.label_to_gen_indices[label][gen].copy()
                random.shuffle(gen_indices_copy[label][gen])

        gen_pointers = {
            label: {gen: 0 for gen in gen_indices_copy[label]}
            for label in self.labels
        }

        for batch_idx in range(self.num_batches):
            batch = []

            for label in self.labels:
                gens = list(gen_indices_copy[label].keys())
                random.shuffle(gens)

                picked = 0
                gen_cycle = 0
                cycles_since_progress = 0

                while picked < self.samples_per_label[label]:
                    gen = gens[gen_cycle % len(gens)]
                    indices = gen_indices_copy[label][gen]
                    ptr = gen_pointers[label][gen]

                    if ptr >= len(indices):
                        if self.cycle_short_labels:
                            # Wrap: reshuffle this generator and rewind.
                            random.shuffle(indices)
                            gen_pointers[label][gen] = 0
                            ptr = 0
                        else:
                            gen_cycle += 1
                            cycles_since_progress += 1
                            if cycles_since_progress >= len(gens):
                                # All generators in this label exhausted.
                                break
                            continue

                    batch.append(indices[ptr])
                    gen_pointers[label][gen] += 1
                    picked += 1
                    gen_cycle += 1
                    cycles_since_progress = 0

            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        return self.num_batches
    
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
                gen_name = os.path.basename(os.path.dirname(item['audio']))
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