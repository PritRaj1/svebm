import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import TextDataModule
from src.collate import TextCollator, create_text_collator


class MockDataset(Dataset):
    def __init__(self, size=100, dim=768):
        self.size = size
        self.dim = dim
        self.data = torch.randn(size, dim)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class MockTextDataset(Dataset):
    """Like IMDB."""

    def __init__(self, size=100):
        self.size = size
        self.texts = [
            f"This is sample text number {i}. It contains some words for testing."
            for i in range(size)
        ]
        self.labels = [i % 2 for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}


@pytest.fixture
def mock_dataset_cls():
    return MockDataset


@pytest.fixture
def mock_text_dataset_cls():
    return MockTextDataset


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
            TextDataModule(
                dataset_cls=mock_dataset_cls,
                dataset_kwargs={"size": 100, "dim": 768},
                val_split=-0.1,
                test_split=0.1,
            )

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

    def test_text_collate_function(self, mock_text_dataset_cls):
        collator = create_text_collator(
            tokenizer_name="bert-base-uncased",
            max_length=64,
            embed_dim=768,
            num_latent_samples=4,
            num_gmm_components=3,
        )

        datamodule = TextDataModule(
            dataset_cls=mock_text_dataset_cls,
            dataset_kwargs={"size": 20},
            batch_size=4,
            collate_fn=collator,
            val_split=0.2,
            test_split=0.2,
        )

        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()

        batch = next(iter(train_loader))

        assert "encoder_inputs" in batch
        assert "inputs" in batch
        assert "targets" in batch
        assert "tgt_probs" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        batch_size = batch["encoder_inputs"].shape[0]
        seq_len = batch["encoder_inputs"].shape[1]
        embed_dim = batch["encoder_inputs"].shape[2]

        assert batch_size > 0
        assert seq_len > 0 and seq_len <= 512  # BERT max length
        assert embed_dim == 768  # BERT base embedding dimension

        assert batch["encoder_inputs"].shape == (batch_size, seq_len, embed_dim)
        assert batch["inputs"].shape == (batch_size, seq_len)
        assert batch["targets"].shape == (batch_size, seq_len)
        assert batch["tgt_probs"].shape == (
            batch_size,
            4,
            3,
        )
        assert batch["attention_mask"].shape == (batch_size, seq_len)
        assert batch["labels"].shape == (batch_size,)


class TestTextCollator:

    def test_collator_instantiation(self):
        collator = TextCollator(
            tokenizer_name="bert-base-uncased",
            max_length=32,
            embed_dim=512,
            num_latent_samples=8,
            num_gmm_components=4,
        )

        assert collator.max_length == 32
        assert collator.embed_dim == 512
        assert collator.num_latent_samples == 8
        assert collator.num_gmm_components == 4
        assert collator.tokenizer is not None

    def test_collator_call(self):
        collator = TextCollator(
            tokenizer_name="bert-base-uncased",
            max_length=16,
            embed_dim=128,
            num_latent_samples=2,
            num_gmm_components=2,
        )

        batch = [
            {"text": "Hello world", "label": 0},
            {"text": "This is a test", "label": 1},
        ]

        result = collator(batch)

        assert isinstance(result, dict)
        assert "encoder_inputs" in result
        assert "inputs" in result
        assert "targets" in result
        assert "tgt_probs" in result

        batch_size, seq_len, embed_dim = result["encoder_inputs"].shape
        assert batch_size == 2
        assert seq_len <= 16
        assert embed_dim == 128
        assert result["inputs"].shape == (2, seq_len)
        assert result["targets"].shape == (2, seq_len)
        assert result["tgt_probs"].shape == (
            2,
            2,
            2,
        )

        assert result["encoder_inputs"].dtype == torch.float32
        assert result["inputs"].dtype == torch.long
        assert result["targets"].dtype == torch.long
        assert result["tgt_probs"].dtype == torch.float32

        assert torch.allclose(result["tgt_probs"].sum(dim=-1), torch.ones(2, 2))
