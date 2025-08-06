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

        else:
            if gen_type == "greedy":
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
            else:
                memory = memory.permute(1, 0, 2)
                return self.beam_search(batch_size, memory, beam_width)

    def beam_search(self, batch_size: int, memory: torch.Tensor, beam_width: int = 3) -> torch.Tensor:
        """
        Beam search to generate and select sequences.
        
        Args:
            batch_size: Number of sequences to generate
            memory: Encoder memory [batch_size, seq_len, embed_size]
            beam_width: Number of beams to maintain
            
        Returns:
            Generated sequences [batch_size, max_seq_len]
        """
        device = memory.device
        max_len = self.max_dec_len
        
        # Initialize beams for each batch item
        beams = [[([self.bos_id], 0.0)] for _ in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]
        
        for step in range(max_len):
            if all(len(beams[b]) == 0 for b in range(batch_size)):
                break
                
            for batch_idx in range(batch_size):
                if len(beams[batch_idx]) == 0:
                    continue
                    
                candidates = []
                
                for seq, log_prob in beams[batch_idx]:
                    seq_tensor = torch.tensor([seq], device=device, dtype=torch.long)
                    memory_batch = memory[:, batch_idx:batch_idx+1]
                    embedded = self.embedding(seq_tensor)
                    
                    if self.concat_latent:
                        latent = self.latent_connector(memory_batch.mean(dim=1))  # [1, embed_size]
                        embedded = embedded + latent.unsqueeze(1)
                    
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_tensor.size(1)).to(device)
                    
                    dec_out = self.transformer(
                        tgt=embedded.permute(1, 0, 2),
                        memory=memory_batch,
                        tgt_mask=causal_mask,
                        tgt_is_causal=True
                    ).permute(1, 0, 2)
                    
                    logits = self.output_layer(dec_out[:, -1, :])
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    top_log_probs, top_indices = log_probs[0].topk(beam_width)
                    
                    for i in range(beam_width):
                        new_token = top_indices[i].item()
                        new_log_prob = top_log_probs[i].item()
                        
                        new_seq = seq + [new_token]
                        new_score = log_prob + new_log_prob
                        
                        candidates.append((new_seq, new_score))
                
                # Select top 'beam_width' candidates
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                
                # Update beams
                new_beams = []
                for seq, log_prob in candidates:
                    if seq[-1] == self.eos_id:
                        finished_beams[batch_idx].append((seq, log_prob))
                    else:
                        new_beams.append((seq, log_prob)) # Continue sequence
                
                beams[batch_idx] = new_beams
        
        outputs = []
        for batch_idx in range(batch_size):
            all_beams = beams[batch_idx] + finished_beams[batch_idx]
            
            if len(all_beams) == 0:
                best_seq = [self.pad_id] * max_len # If no valid sequences, return padding
            else:
                best_seq, _ = max(all_beams, key=lambda x: x[1]) # Select with highest log probability
                
                if len(best_seq) < max_len:
                    best_seq.extend([self.pad_id] * (max_len - len(best_seq)))
                else:
                    best_seq = best_seq[:max_len]
            
            outputs.append(best_seq)
        
        return torch.tensor(outputs, device=device, dtype=torch.long)
