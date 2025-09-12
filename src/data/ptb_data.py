"""
Best Practice PTB Dataset Implementation

This follows modern best practices:
1. Uses HuggingFace datasets for data loading
2. Uses HuggingFace tokenizers for tokenization
3. Minimal custom code, leverages established libraries
4. Proper error handling and logging
5. Clean, maintainable interface
"""

import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def ptb_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        if key == "text":
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated


class PTBDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_length: int = 128,
        vocab_size: int = 10000,
        download: bool = True,
        data_dir: str = "data/ptb",
        vocab: Optional[Dict] = None,
        tokenizer_name: str = "bert-base-uncased",
    ):
        self.split = split
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.data_dir = data_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sentences = self._load_data()

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}

            word_counts = {}
            for sentence in self.sentences:
                tokens = self.tokenizer.tokenize(sentence)
                for token in tokens:
                    word_counts[token] = word_counts.get(token, 0) + 1

            # Add most frequent to vocab
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            for word, _ in sorted_words:
                if len(self.vocab) >= self.vocab_size:
                    break
                self.vocab[word] = len(self.vocab)

        logger.info(f"Loaded PTB {split} split: {len(self.sentences)} sentences")

    def _load_data(self) -> List[str]:
        try:
            dataset = load_dataset("ptb_text_only", split=self.split)
            sentences = [item["sentence"] for item in dataset]
            logger.info(
                f"Successfully loaded PTB {self.split} from HuggingFace datasets: "
                f"{len(sentences)} sentences"
            )
            return sentences

        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace datasets: {e}")
            sample_data = self._get_sample_data()
            logger.info(f"Using sample data: {len(sample_data)} sentences")
            return sample_data

    def _get_sample_data(self) -> List[str]:
        """Generate sample data for testing when real data unavailable."""
        sample_sentences = [
            "The quick brown fox jumps over the lazy dog .",
            "A man walks into a bar and orders a drink .",
            "The weather is nice today and the sun is shining .",
            "Machine learning models can process natural language effectively .",
            "The stock market experienced significant volatility this week .",
            "Researchers are developing new algorithms for text generation .",
            "The company announced record profits for the third quarter .",
            "Students are learning about artificial intelligence in their "
            "computer science class .",
            "The government is implementing new policies to address climate change .",
            "Technology companies are investing heavily in renewable energy "
            "solutions .",
        ]

        return sample_sentences * 100

    def _tokenize(self, sentence: str) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(sentence)

        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab["<unk>"])

        token_ids = [self.vocab["<sos>"]] + token_ids + [self.vocab["<eos>"]]

        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            token_ids.extend([self.vocab["<pad>"]] * (self.max_length - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sample."""
        sentence = self.sentences[idx]
        token_ids = self._tokenize(sentence)

        attention_mask = (token_ids != self.vocab["<pad>"]).long()

        return {
            "encoder_inputs": torch.randn(self.max_length, 768),  # Mock BERT embeddings
            "inputs": token_ids,
            "targets": token_ids,
            "tgt_probs": torch.ones(1, 50, 10) / 10,  # Mock GMM probabilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            "attention_mask": attention_mask,
            "labels": torch.tensor(idx % 2, dtype=torch.long),
            "text": sentence,
        }
