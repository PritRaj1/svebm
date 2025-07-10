import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from src.data import TextDataModule


class MockDataset(Dataset):
    def __init__(self, size=100, dim=768):
        self.size = size
        self.dim = dim
        self.data = torch.randn(size, dim)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def mock_dataset_cls():
    return MockDataset


class TestTextDataModule:
    def test_init(self, mock_dataset_cls):
        datamodule = TextDataModule(
            dataset_cls=mock_dataset_cls,
            dataset_kwargs={"size": 100, "dim": 768},
            batch_size=32,
            num_workers=2,
            val_split=0.1,
            test_split=0.1,
        )

        assert datamodule.dataset_cls == mock_dataset_cls
        assert datamodule.batch_size == 32
        assert datamodule.val_split == 0.1
        assert datamodule.test_split == 0.1

    def test_invalid_init(self, mock_dataset_cls):
        with pytest.raises(
            ValueError, match="val_split and test_split must be non-negative"
        ):
            TextDataModule(dataset_cls=mock_dataset_cls, val_split=-0.1, test_split=0.1)

    def test_fit_split(self, mock_dataset_cls):
        datamodule = TextDataModule(
            dataset_cls=mock_dataset_cls,
            dataset_kwargs={"size": 100, "dim": 768},
            val_split=0.1,
            test_split=0.1,
        )

        datamodule.setup(stage="fit")

        assert datamodule.data_train is not None
        assert datamodule.data_val is not None
        assert datamodule.data_test is not None
        assert len(datamodule.data_train) == 80  # 100 - 10 - 10
        assert len(datamodule.data_val) == 10
        assert len(datamodule.data_test) == 10

    def test_dataloaders(self, mock_dataset_cls):
        datamodule = TextDataModule(
            dataset_cls=mock_dataset_cls,
            dataset_kwargs={"size": 100, "dim": 768},
            batch_size=32,
            val_split=0.1,
            test_split=0.1,
        )
        datamodule.setup(stage="fit")

        train_loader = datamodule.train_dataloader()
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 32

        val_loader = datamodule.val_dataloader()
        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 32

        test_loader = datamodule.test_dataloader()
        assert isinstance(test_loader, DataLoader)
        assert test_loader.batch_size == 32

    def test_uninit_datalaoder(self, mock_dataset_cls):
        datamodule = TextDataModule(
            dataset_cls=mock_dataset_cls, dataset_kwargs={"size": 100, "dim": 768}
        )

        with pytest.raises(RuntimeError, match="Dataset not set up"):
            datamodule.train_dataloader()

    def test_autodim(self, mock_dataset_cls):
        datamodule = TextDataModule(
            dataset_cls=mock_dataset_cls,
            dataset_kwargs={"size": 100, "dim": 512},
            auto_detect_dim=True,
        )

        datamodule.setup(stage="fit")
        assert datamodule.data_dim == 512

    def test_reproducible_splits(self, mock_dataset_cls):
        datamodule1 = TextDataModule(
            dataset_cls=mock_dataset_cls,
            dataset_kwargs={"size": 100, "dim": 768},
            seed=42,
        )

        datamodule2 = TextDataModule(
            dataset_cls=mock_dataset_cls,
            dataset_kwargs={"size": 100, "dim": 768},
            seed=42,
        )

        datamodule1.setup(stage="fit")
        datamodule2.setup(stage="fit")

        assert len(datamodule1.data_train) == len(datamodule2.data_train)
        assert len(datamodule1.data_val) == len(datamodule2.data_val)
        assert len(datamodule1.data_test) == len(datamodule2.data_test)
