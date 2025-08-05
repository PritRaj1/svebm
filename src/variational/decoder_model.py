from typing import Any, cast

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
        hidden_layers: list = [128, 128],
        nhead: int = 4,
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
        self.nhead = nhead
        self.dropout = dropout
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.max_dec_len = max_dec_len
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = (
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        )
        self.concat_latent = concat_latent

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)

        # Transformer decoder layers
        decoder_layers = []
        for _ in range(len(hidden_layers)):
            decoder_layers.append(
                nn.TransformerDecoderLayer(
                    d_model=embed_size,
                    nhead=nhead,
                    dim_feedforward=hidden_layers[-1],
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_size,
                nhead=nhead,
                dim_feedforward=hidden_layers[-1],
                dropout=dropout,
                activation=activation,
            ),
            num_layers=len(hidden_layers),
        )

        # Output projection to vocabulary
        self.output_layer = nn.Linear(embed_size, vocab_size)

        # Latent space connector
        self.latent_connector = (
            nn.Linear(latent_dim, embed_size) if concat_latent else None
        )

    def forward(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        mode: str = "TEACH_FORCE",
        gen_type: str = "greedy",
    ) -> torch.Tensor:
        """
        Decode latent space to tokens.

        Args:
            inputs: Input token sequences (batch, seq_len) for teacher forcing
            memory: Encoder memory/latent representations (batch, seq_len, embed_size)
            mode: 'TEACH_FORCE' for training, 'GENERATE' for inference
            gen_type: 'greedy' or 'beam' for generation

        Returns:
            Logits (batch, seq_len, vocab_size) or generated sequences
        """
        batch_size = inputs.size(0) if inputs is not None else memory.size(0)
        device = memory.device

        # Learn by teacher forcing
        if mode == "TEACH_FORCE":
            embedded = self.embedding(inputs)

            if self.concat_latent:
                latent = self.latent_connector(memory)
                embedded = embedded + latent.unsqueeze(1)

            seq_len = inputs.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            dec_out = self.transformer(embedded, memory=memory, tgt_mask=causal_mask, tgt_is_causal=True)
            logits = self.output_layer(dec_out)
            return logits

        # Inference pass
        else:
            if gen_type == "greedy":
                current_input = torch.full(
                    (batch_size, 1), self.bos_id, device=device, dtype=torch.long
                )
                outputs = []

                for _ in range(self.max_dec_len):
                    embedded = self.embedding(current_input)

                    if self.concat_latent:
                        latent = self.latent_connector(memory)
                        embedded = embedded + latent.unsqueeze(1)

                    # Create causal mask for current sequence length
                    current_seq_len = current_input.size(1)
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(current_seq_len).to(device)

                    dec_out = self.transformer(
                        embedded, memory=memory, tgt_mask=causal_mask, tgt_is_causal=True
                    )

                    logits = self.output_layer(dec_out[:, -1, :])

                    next_token = logits.argmax(dim=-1, keepdim=True)

                    outputs.append(next_token)
                    current_input = torch.cat([current_input, next_token], dim=1)

                    # Stop if all sequences have EOS token
                    if (next_token == self.eos_id).all():
                        break

                return torch.cat(outputs, dim=1)
            else:
                return self.beam_search(batch_size, memory)

    def beam_search(
        self,
        batch_size: int,
        memory: torch.Tensor,
        beam_width: int = 3,
        max_len: int = None,
    ) -> torch.Tensor:
        """
        Beam search for inference; select most likely sequence.

        Args:
            batch_size: Number of sequences to generate
            memory: Encoder memory/latent representations (batch, seq_len, embed_size)
            beam_width: Number of beams to maintain
            max_len: Maximum sequence length (defaults to self.max_dec_len)

        Returns:
            Generated sequences (batch_size, seq_len)
        """
        device = memory.device
        max_len = max_len or self.max_dec_len

        # Initialize beam with BOS token - fix the list creation
        beams = [[([self.bos_id], 0.0)] for _ in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            all_candidates = []
            for b in range(batch_size):
                candidates = []
                for seq, log_prob in beams[b]:
                    current_input = torch.tensor(
                        [seq], device=device, dtype=torch.long
                    )  # [1, seq_len]
                    embedded = self.embedding(current_input)
                    if self.concat_latent:
                        latent = self.latent_connector(
                            memory[b : b + 1]
                        )  # [1, embed_size]
                        embedded = embedded + latent.unsqueeze(1)
                    
                    # Create causal mask for current sequence length
                    current_seq_len = current_input.size(1)
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(current_seq_len).to(device)
                    
                    dec_out = self.transformer(
                        embedded, memory=memory[b : b + 1], tgt_mask=causal_mask, tgt_is_causal=True
                    )
                    logits = self.output_layer(dec_out[:, -1, :])
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Get top 'beam_width' tokens
                    top_log_probs, top_indices = log_probs[0].topk(beam_width)
                    for i in range(beam_width):
                        new_seq = seq + [top_indices[i].item()]
                        new_log_prob = log_prob + top_log_probs[i].item()
                        candidates.append((new_seq, new_log_prob))

                # Select top 'beam_width' candidates
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[
                    :beam_width
                ]
                new_beams = []
                for seq, log_prob in candidates:
                    if seq[-1] == self.eos_id:
                        finished_beams[b].append((seq, log_prob))
                    else:
                        new_beams.append((seq, log_prob))
                beams[b] = new_beams if new_beams else finished_beams[b][:beam_width]

            if all(len(beams[b]) == 0 for b in range(batch_size)):
                break

        # Return best per batch
        outputs = []
        for b in range(batch_size):
            all_sequences = beams[b] + finished_beams[b]
            best_seq = (
                max(all_sequences, key=lambda x: x[1])[0]
                if all_sequences
                else [self.pad_id] * max_len
            )
            outputs.append(best_seq)
        return torch.tensor(
            outputs, device=device, dtype=torch.long
        )
