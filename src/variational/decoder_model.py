from typing import Any
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import get_activation

class DecoderModel(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        latent_dim: int,
        memory_dim: int,
        hidden_layers: list = [128, 128],
        nhead: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
        max_dec_len: int = 50,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        unk_id: int = 3,
        concat_latent: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.latent_dim = latent_dim
        self.memory_dim = memory_dim
        self.nhead = nhead
        self.dropout = dropout
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.max_dec_len = max_dec_len
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = pad_id, bos_id, eos_id, unk_id
        self.concat_latent = concat_latent

        # Embedding and projection layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.latent_projection = nn.Linear(latent_dim, embed_size)
        self.memory_projection = nn.Linear(memory_dim, embed_size)

        # Transformer decoder
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_size,
                nhead=nhead,
                dim_feedforward=hidden_layers[-1],
                dropout=dropout,
                activation=activation,
            ),
            num_layers=len(hidden_layers)
        )

        # Output projection
        self.output_layer = nn.Linear(embed_size, vocab_size)

        # Latent connector for conditioning (optional)
        self.latent_connector = nn.Linear(latent_dim, embed_size) if concat_latent else None

    def forward(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        mode: str = "TEACH_FORCE",
        gen_type: str = "greedy",
        beam_width: int = 3,
    ) -> torch.Tensor:
        batch_size = inputs.size(0) if inputs is not None else memory.size(0)
        device = memory.device
        memory = self.memory_projection(memory)

        if mode == "TEACH_FORCE":
            embedded = self.embedding(inputs)
            
            if self.concat_latent:
                latent = self.latent_connector(memory.mean(dim=1))
                embedded = embedded + latent.unsqueeze(1)
            
            causal_mask = nn.Transformer.generate_square_subsequent_mask(inputs.size(1)).to(device)
            dec_out = self.transformer(
                tgt=embedded.permute(1, 0, 2),
                memory=memory.permute(1, 0, 2),
                tgt_mask=causal_mask,
                tgt_is_causal=True
            ).permute(1, 0, 2)
            return self.output_layer(dec_out)

        else:  # GENERATE mode
            outputs = []
            current_input = torch.full((batch_size, 1), self.bos_id, device=device, dtype=torch.long)
            memory = memory.permute(1, 0, 2)

            for _ in range(self.max_dec_len):
                embedded = self.embedding(current_input)
                
                if self.concat_latent:
                    latent = self.latent_connector(memory.mean(dim=0))
                    embedded = embedded + latent.unsqueeze(1)
                
                causal_mask = nn.Transformer.generate_square_subsequent_mask(current_input.size(1)).to(device)
                dec_out = self.transformer(
                    tgt=embedded.permute(1, 0, 2),
                    memory=memory,
                    tgt_mask=causal_mask,
                    tgt_is_causal=True
                ).permute(1, 0, 2)
                logits = self.output_layer(dec_out[:, -1, :])

                next_token = logits.argmax(dim=-1, keepdim=True)

                outputs.append(next_token)
                current_input = torch.cat([current_input, next_token], dim=1)
                if (next_token == self.eos_id).all():
                    break

            return torch.cat(outputs, dim=1)

    def beam_search(self, batch_size: int, memory: torch.Tensor, beam_width: int = 3) -> torch.Tensor:
        """
        Greedy beam search.

        Args:
            batch_size (int): Batch size.
            memory (torch.Tensor): Memory tensor (batch, seq_len, memory_dim).
            beam_width (int): Beam width.

        Returns:
            torch.Tensor: Decoded sequences (batch, max_dec_len).
        """
        device = memory.device
        max_len = self.max_dec_len

        outputs = []
        for b in range(batch_size):
            current_input = torch.full((1, 1), self.bos_id, device=device, dtype=torch.long)
            memory_b = memory[:, b:b+1, :]
            
            seq = [self.bos_id]
            for _ in range(max_len):
                embedded = self.embedding(current_input)
                if self.concat_latent:
                    latent = self.latent_connector(memory_b.mean(dim=0))
                    embedded = embedded + latent.unsqueeze(1)
                
                causal_mask = nn.Transformer.generate_square_subsequent_mask(current_input.size(1)).to(device)
                dec_out = self.transformer(
                    tgt=embedded.permute(1, 0, 2),
                    memory=memory_b,
                    tgt_mask=causal_mask,
                    tgt_is_causal=True
                ).permute(1, 0, 2)
                logits = self.output_layer(dec_out[:, -1, :])
                
                next_token = logits.argmax(dim=-1, keepdim=True)
                seq.append(next_token.item())
                current_input = torch.cat([current_input, next_token], dim=1)
                
                if next_token.item() == self.eos_id:
                    break
            
            outputs.append(seq)
        
        max_seq_len = max(len(seq) for seq in outputs)
        padded_outputs = []
        for seq in outputs:
            if len(seq) < max_seq_len:
                seq.extend([self.pad_id] * (max_seq_len - len(seq)))
            padded_outputs.append(seq[:max_seq_len])
        
        return torch.tensor(padded_outputs, device=device, dtype=torch.long)
