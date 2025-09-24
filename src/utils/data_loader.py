import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pathlib import Path
from typing import Any, Optional, List, Tuple, Union
from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset
import torch
from src.utils.arguments import get_args
from src.base_pipeline import BasePipeline
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ArtMancerDataLoader:
    def __init__(self, dataset_dir: str, auto_download: bool = True, hf_dataset_name: Optional[str] = "Ekanari/Adobe_5K", pipeline = None):
        """
        Initialize the ArtMancer DataLoader
        Args:
            dataset_dir (str): Directory where the dataset is stored.
            auto_download (bool): Whether to automatically download the dataset if not found.
            hf_dataset_name (str): HuggingFace dataset name
        """
        self.dataset_dir = Path(dataset_dir)
        self.raw_dir = self.dataset_dir / "raw"
        self.processed_dir = self.dataset_dir / "processed"
        self.auto_download = auto_download
        self.dataset_name = hf_dataset_name
        self.pipeline = pipeline

        # Initialize dataset splits
        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Create directories if they do not exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load or download dataset
        self._initialize_dataset()

    def _initialize_dataset(self):
        """
        Initialize dataset - load from cache or download if not available.
        """
        try:
            # Check if dataset is already downloaded
            if self._has_processed_data():
                print(f"Loading dataset from {self.processed_dir}...")
                self._load_from_processed()

            # Check if raw dataset exists
            elif self._has_raw_data():
                print(f"Found raw dataset in {self.raw_dir}. Loading...")
                self._load_from_raw()
                self._create_splits()
                self._save_processed()

            # Download dataset if not found
            elif self.auto_download:
                print("Downloading dataset...")
                self.download_dataset()

            else:
                raise FileNotFoundError(
                    f"Dataset not found in {self.dataset_dir}. "
                    f"Set auto_download=True or run download_dataset()."
                )
        except Exception as e:
            print(f"Error initializing dataset: {e}")
            raise e

    def _has_processed_data(self) -> bool:
        """Check if processed data exists."""
        required_files = ["train.parquet",
                          "validation.parquet", "test.parquet"]
        return all((self.processed_dir / file).exists() for file in required_files)

    def _has_raw_data(self) -> bool:
        """Check if raw data exists."""
        cache_indicators = ["dataset_info.json", "state.json"]
        return any(list(self.raw_dir.rglob(indicator)) for indicator in cache_indicators)

    def _load_from_processed(self):
        """Load dataset from processed parquet files."""
        try:
            # Load all parquet files
            files = {
                'train': self.processed_dir / "train.parquet",
                'validation': self.processed_dir / "validation.parquet",
                'test': self.processed_dir / "test.parquet"
            }

            dataframes = {}
            for split, file_path in files.items():
                if file_path.exists():
                    dataframes[split] = pd.read_parquet(file_path)
                else:
                    raise FileNotFoundError(
                        f"Processed file {file_path} not found.")

            # Convert to Hugging Face
            self.train_data = HFDataset.from_pandas(
                dataframes['train']) if 'train' in dataframes else None
            self.val_data = HFDataset.from_pandas(
                dataframes['validation']) if 'validation' in dataframes else None
            self.test_data = HFDataset.from_pandas(
                dataframes['test']) if 'test' in dataframes else None

            print("Loaded processed dataset:")
            print(
                f"Train samples: {len(self.train_data) if self.train_data else 0}")
            print(
                f"Validation samples: {len(self.val_data) if self.val_data else 0}")
            print(
                f"Test samples: {len(self.test_data) if self.test_data else 0}")

        except Exception as e:
            print(f"Failed to load processed dataset: {e}")
            print("Falling back to raw dataset loading...")
            self._load_from_raw()

    def _load_from_raw(self):
        """Load dataset from raw Hugging Face cache."""
        try:
            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=str(self.raw_dir),
            )
            print(
                f"Loaded raw dataset with samples.")

        except Exception as e:
            print(f"Failed to load raw dataset: {e}")
            if self.auto_download:
                self.download_dataset()
            else:
                raise FileNotFoundError(
                    f"Dataset {self.dataset_name} not found in {self.raw_dir}. "
                    f"Set auto_download=True or run download_dataset()."
                )

    def download_dataset(self, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42) -> None:
        """Download and prepare dataset from Hugging Face."""
        self.seed = seed

        try:
            if self.dataset_name is None:
                raise ValueError(
                    "Dataset name must be specified for downloading.")

            print("Downloading dataset...")

            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=str(self.raw_dir),
            )

            print(f"Downloaded dataset {self.dataset_name} to {self.raw_dir}")

            # Check dataset structure
            if "train" in self.dataset:
                full_dataset = self.dataset["train"]
                print("Total samples in train set:", len(full_dataset))
                if len(full_dataset) > 0:
                    sample = full_dataset[0]
                    print(f"Sample keys: {list(sample.keys())}")

                # Create splits and save processed data
                self._create_splits(test_size, val_size, seed)
                self._save_processed()
            else:
                raise ValueError(
                    f"Dataset {self.dataset_name} does not contain 'train' split.")

        except Exception as e:
            print(f"Failed to download dataset {self.dataset_name}: {e}")
            raise

    def _create_splits(self, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
        """Create train, validation, and test splits."""
        if not self.dataset or "train" not in self.dataset:
            raise ValueError(
                "Dataset not loaded. Please download or load the dataset first.")

        full_dataset = self.dataset["train"]

        # Create train/test split
        train_test_split = full_dataset.train_test_split(
            test_size=test_size, seed=seed)

        # Create validation split from remaining train data
        remaining_train_size = 1 - test_size
        val_size_adjusted = val_size / remaining_train_size

        train_val_split = train_test_split["train"].train_test_split(
            test_size=val_size_adjusted, seed=seed)

        # Train split
        self.train_data = train_val_split["train"]

        # Validation split
        self.val_data = train_val_split["test"]

        # Test split
        self.test_data = train_test_split["test"]

        print("Created splits:")
        print(
            f"   Train: {len(self.train_data)} samples ({len(self.train_data)/len(full_dataset)*100:.1f}%)")
        print(
            f"   Validation: {len(self.val_data)} samples ({len(self.val_data)/len(full_dataset)*100:.1f}%)")
        print(
            f"   Test: {len(self.test_data)} samples ({len(self.test_data)/len(full_dataset)*100:.1f}%)")

    def _save_processed(self):
        """Save processed splits as parquet files."""
        try:
            print("Saving processed dataset to parquet files...")

            # Save splits with consistent naming
            splits = {
                'train': self.train_data,
                'validation': self.val_data,
                'test': self.test_data
            }

            for split_name, split_data in splits.items():
                if split_data:
                    file_path = self.processed_dir / f"{split_name}.parquet"
                    split_data.to_parquet(str(file_path))

            # Save metadata
            metadata = {
                "dataset_name": self.dataset_name,
                "total_samples": len(self.dataset["train"]) if self.dataset else 0,
                "train_samples": len(self.train_data) if self.train_data else 0,
                "val_samples": len(self.val_data) if self.val_data else 0,
                "test_samples": len(self.test_data) if self.test_data else 0,
                "seed": getattr(self, 'seed', 42),
                "created_at": str(pd.Timestamp.now()),
                "splits_ratio": {
                    "train": len(self.train_data) / len(self.dataset["train"]) if self.dataset and self.train_data else 0,
                    "val": len(self.val_data) / len(self.dataset["train"]) if self.dataset and self.val_data else 0,
                    "test": len(self.test_data) / len(self.dataset["train"]) if self.dataset and self.test_data else 0,
                }
            }

            with open(self.processed_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print("Processed dataset saved successfully.")

        except Exception as e:
            print(f"Failed to save processed dataset: {e}")

    

    def create_condition_gt_pairs(self, sample: dict) -> List[Tuple[Any, Any]]:
        """
        Create pairs [condition-ground truth] from sample
        Args:
            sample: Dict contains img1, img2, img3, and gt (ground truth)
        Returns:
            List of tuples: [(condition1, gt1), (condition2, gt2), ...]
        """
        
        pairs = []
        gt = sample.get('gt')

        if gt is None:
            print("Sample must contain 'gt' key for ground truth.")
            return pairs

        #gt = self.pipeline.forward_vae(input=gt, mode='gt')

        for img_key in ['img1', 'img2', 'img3']:
            if img_key in sample:
                condition = sample[img_key]
                #condition = self.pipeline.forward_vae(input=condition, mode=['condition'])
                pairs.append((condition, gt))

        return pairs

    def stack_pairs(self, pairs: List[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any]]:
        """
        Stack pairs of conditions and ground truths
        Args:
            pairs: List of tuples [(condition1, gt1), (condition2, gt2), ...]
        Returns:
            Tuple of lists: (stacked_conditions, stacked_ground_truths)
        """
        if not pairs:
            return [], []

        conditions = [pair[0] for pair in pairs]
        ground_truths = [pair[1] for pair in pairs]

        return conditions, ground_truths


    def create_null_encoder_hidden_states(self, num_pairs: int, hidden_dim: int = 768) -> torch.Tensor:
        """
        Create null encoder hidden states for the given number of pairs
        Args:
            num_pairs: Number of pairs to create hidden states for
            hidden_dim: Dimension of the hidden states (default is 768)
        Returns:
            torch.Tensor: Null hidden states of shape (num_pairs, hidden_dim)
        """
        if num_pairs <= 0:
             raise ValueError("Number of pairs must be greater than 0")
        #null_conditioning = self.pipeline.text_encoder(self.pipeline.tokenize_captions([""]).to(self.device))[0]
        return torch.zeros(num_pairs, hidden_dim)
        #return null_conditioning

    def __len__(self) -> int:
        return len(self.train_data) if self.train_data else 0

    def __getitem__(self, idx: int) -> dict:
        """
        Get sample by requested process
        Returns:
            Dict contains:
                - conditions: List of conditions (images)
                - ground_truths: List of ground truths
                - null_encoder_hidden_states: Null hidden states for encoder
        """
        if not self.train_data:
            raise RuntimeError("No training data available")

        if idx < 0 or idx >= len(self.train_data):
            raise IndexError(f"Index {idx} out of range")
        
        sample = self.train_data[idx]

        # Create pairs [condition-ground truth]
        pairs = self.create_condition_gt_pairs(sample)

        # Stack pairs into conditions and ground truths
        conditions, ground_truths = self.stack_pairs(pairs)

        # Create null encoder hidden states
        null_encoder_hidden_states = self.create_null_encoder_hidden_states(
            num_pairs=len(pairs), hidden_dim=768)

        return {
            "conditions": conditions,
            "ground_truths": ground_truths,
            "null_encoder_hidden_states": null_encoder_hidden_states,
            "num_pairs": len(pairs),
            "original_sample": sample
        }

    def get_train_data(self) -> Optional[HFDataset]:
        return self.train_data

    def get_test_data(self) -> Optional[HFDataset]:
        return self.test_data

    def get_val_data(self) -> Optional[HFDataset]:
        return self.val_data

    def get_splits(self) -> Tuple['ArtMancerDataSplit', 'ArtMancerDataSplit', 'ArtMancerDataSplit']:
        """Get train, validation, and test splits."""
        train_dataset = ArtMancerDataSplit(self.train_data, self)
        val_dataset = ArtMancerDataSplit(self.val_data, self)
        test_dataset = ArtMancerDataSplit(self.test_data, self)
        return train_dataset, val_dataset, test_dataset


class ArtMancerDataSplit(Dataset):
    """Wrapper for ArtMancer dataset splits."""

    def __init__(self, data: Optional[HFDataset], parent_dataset: ArtMancerDataLoader):
        self.data = data
        self.parent = parent_dataset

    def __len__(self):
        return len(self.data) if self.data else 0

    def __getitem__(self, idx) -> dict:
        if not self.data:
            raise RuntimeError("No data available in this split")

        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range")

        sample = self.data[idx]
        pairs = self.parent.create_condition_gt_pairs(sample)
        conditions, ground_truths = self.parent.stack_pairs(pairs)
        null_hidden_states = self.parent.create_null_encoder_hidden_states(
            num_pairs=len(pairs), hidden_dim=768
        )

        return {
            'conditions': conditions,
            'ground_truths': ground_truths,
            'null_hidden_states': null_hidden_states,
            'num_pairs': len(pairs),
            'original_sample': sample
        }


if __name__ == "__main__":
    try:
        args = get_args()
        pipeline = BasePipeline(args=args)
        pipeline.load_pretrained_tokenizer()
        pipeline.load_pretrained_text_encoder()
        pipeline.load_pretrained_vae()
        loader = ArtMancerDataLoader(
            "./data", auto_download=True, pipeline=pipeline)  # Fixed class name

        print(f"\nDataset info:")
        if loader.dataset:
            print(f"Original dataset splits: {list(loader.dataset.keys())}")

        train_data = loader.get_train_data()
        print(f"Train samples: {len(train_data) if train_data else 0}")

        if train_data and len(train_data) > 0:
            # Test sample structure
            sample = train_data[0]
            print(f"Sample keys: {list(sample.keys())}")

            # Test PyTorch dataset wrapper
            train_ds, val_ds, test_ds = loader.get_splits()
            print(
                f"PyTorch datasets - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

            # Test sample processing
            if len(train_ds) > 0:
                processed_sample = train_ds[0]
                print(f"Processed sample keys: {list(processed_sample.keys())}")
                print(f"Number of condition-GT pairs: {processed_sample.get('num_pairs')}")

                # ✅ Check if each key holds a tensor
                for key, value in processed_sample.items():
                    if torch.is_tensor(value):
                        print(f"{key}: tensor {tuple(value.shape)} dtype={value.dtype} range=({value.min().item():.3f}, {value.max().item():.3f})")
                    elif isinstance(value, list):
                        print(f"{key}: list of length {len(value)}")
                        if len(value) > 0 and torch.is_tensor(value[0]):
                            print(f"  first item shape={tuple(value[0].shape)} dtype={value[0].dtype}")
                    else:
                        print(f"{key}: {type(value)}")
        else:
            print("No training data available.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
