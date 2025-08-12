"""Collate functions for text data processing."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any


class TextCollator:
    """Collate function for text data that converts raw text to model inputs."""

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        unk_id: int = 3,
        embed_dim: int = 768,
        num_latent_samples: int = 20,
        num_gmm_components: int = 5,
    ):
        """Initialize text collator.

        Args:
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            pad_id: Padding token ID
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID
            unk_id: Unknown token ID
            embed_dim: Embedding dimension for encoder inputs
            num_latent_samples: Number of latent samples for EBM
            num_gmm_components: Number of GMM components
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.embed_dim = embed_dim
        self.num_latent_samples = num_latent_samples
        self.num_gmm_components = num_gmm_components

        # Set special tokens if tokenizer doesn't have them
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        # Freeze the model to avoid training it
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of raw text samples.

        Args:
            batch: List of samples from dataset, each with 'text' and 'label' keys

        Returns:
            Dictionary with model input tensors
        """
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            embeddings = self.model.embeddings.word_embeddings(input_ids)

            if embeddings.size(-1) != self.embed_dim:
                projection = torch.nn.Linear(embeddings.size(-1), self.embed_dim)
                embeddings = projection(embeddings)

            encoder_inputs = embeddings

        # Add BOS at the beginning, remove last token
        decoder_inputs = torch.full((batch_size, 1), self.bos_id, dtype=torch.long)
        if seq_len > 1:
            decoder_inputs = torch.cat([decoder_inputs, input_ids[:, :-1]], dim=1)
        else:
            decoder_inputs = decoder_inputs  # Just BOS token

        targets = input_ids.clone()
        targets[attention_mask == 0] = self.pad_id # Ignore tokens

        # TODO: GMM target probabilities (uniform distribution for now)
        tgt_probs = torch.ones(
            batch_size,
            self.num_latent_samples,
            self.num_gmm_components,
        )
        tgt_probs = F.softmax(tgt_probs, dim=-1)

        return {
            "encoder_inputs": encoder_inputs,
            "inputs": decoder_inputs,
            "targets": targets,
            "tgt_probs": tgt_probs,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_text_collator(
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
    embed_dim: int = 768,
    num_latent_samples: int = 20,
    num_gmm_components: int = 5,
) -> TextCollator:
    """Create a TextCollator instance with default parameters."""
    return TextCollator(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        embed_dim=embed_dim,
        num_latent_samples=num_latent_samples,
        num_gmm_components=num_gmm_components,
    )
