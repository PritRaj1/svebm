import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional, Callable, Any, Dict, Union
import torch
import logging

logger = logging.getLogger(__name__)

class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cls: Callable[..., Dataset],
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: Union[float, int] = 0.1,
        test_split: Union[float, int] = 0.1,
        collate_fn: Optional[Callable] = None,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        auto_detect_dim: bool = True,
    ):
        super().__init__()
        
        if val_split < 0 or test_split < 0:
            raise ValueError("val_split and test_split must be non-negative")
        
        if isinstance(val_split, float) and isinstance(test_split, float):
            if val_split + test_split >= 1.0:
                raise ValueError("val_split + test_split must be less than 1.0")
        
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs or {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed = seed
        self.auto_detect_dim = auto_detect_dim
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_dim: Optional[int] = None
        
        self.generator = torch.Generator().manual_seed(self.seed)

    def prepare_data(self):
        try:
            self.dataset_cls(**self.dataset_kwargs)
            logger.info(f"Dataset prepared successfully")
        except Exception as e:
            logger.warning(f"Error in prepare_data: {e}")

    def _detect_data_dimension(self, dataset: Dataset) -> int:
        """Parse dimensions from dataset."""
        try:
            sample = dataset[0]
            
            if isinstance(sample, dict):
                if 'input_ids' in sample:
                    if hasattr(dataset, 'features') and 'input_ids' in dataset.features:
                        return dataset.features['input_ids'].feature.shape[-1]
                    else:
                        return len(sample['input_ids'])
                elif 'text' in sample:
                    logger.warning("Raw text detected. Please provide tokenizer in collate_fn for dimension detection.")
                    return 768  # Default BERT dimension
                else:
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            return value.shape[-1]
                        
            elif isinstance(sample, torch.Tensor):
                return sample.shape[-1]
            elif isinstance(sample, (list, tuple)):
                if len(sample) > 0:
                    if isinstance(sample[0], torch.Tensor):
                        return sample[0].shape[-1]
            
            logger.warning("Could not detect data dimension automatically. Using default 768.")
            return 768
            
        except Exception as e:
            logger.warning(f"Error detecting data dimension: {e}. Using default 768.")
            return 768

    def setup(self, stage: Optional[str] = None):
        """Called on every process in Lightning's DDPStrategy."""
        
        # Training splits
        if stage == "fit" or stage is None:
            full_dataset = self.dataset_cls(**self.dataset_kwargs)
            
            if self.auto_detect_dim:
                self.data_dim = self._detect_data_dimension(full_dataset)
                logger.info(f"Auto-detected data dimension: {self.data_dim}")
            
            dataset_len = len(full_dataset)
            val_len = int(self.val_split * dataset_len) if isinstance(self.val_split, float) else self.val_split
            test_len = int(self.test_split * dataset_len) if isinstance(self.test_split, float) else self.test_split
            train_len = dataset_len - val_len - test_len
            
            if train_len <= 0:
                raise ValueError(f"Invalid splits: train_len={train_len}, val_len={val_len}, test_len={test_len}")
            
            if self.shuffle:
                splits = random_split(full_dataset, [train_len, val_len, test_len], generator=self.generator)
            else:
                splits = random_split(full_dataset, [train_len, val_len, test_len])
            
            self.data_train, self.data_val, self.data_test = splits
            
            logger.info(f"Dataset splits - Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")
        
        # Validation splits
        elif stage == "test":
            if self.data_test is None:
                self.setup("fit")
                
        # Full dataset for prediction
        elif stage == "predict":
            self.data_train = self.dataset_cls(**self.dataset_kwargs) 

    def train_dataloader(self):
        if self.data_train is None:
            raise RuntimeError("Dataset not set up. Call setup() first.")
            
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.data_val is None:
            return None
            
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.data_test is None:
            return None
            
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        if self.data_train is None:
            raise RuntimeError("Dataset not set up. Call setup() first.")
            
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )
