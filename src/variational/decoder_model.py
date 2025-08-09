import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


class GenerationMode(Enum):
    TEACH_FORCE = "TEACH_FORCE"
    GREEDY = "greedy"
    BEAM = "beam"


class TokenManager:
    """Manages token operations."""

    def __init__(self, pad_id: int, bos_id: int, eos_id: int, unk_id: int):
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def create_bos_tensor(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create a tensor of BOS tokens."""
        return torch.full((batch_size, 1), self.bos_id, device=device, dtype=torch.long)

    def is_eos(self, tokens: torch.Tensor) -> torch.Tensor:
        """Check if tokens are EOS tokens."""
        return tokens == self.eos_id

    def pad_sequence(self, sequence: List[int], max_len: int) -> List[int]:
        """Pad a sequence to max_len."""
        if len(sequence) < max_len:
            return sequence + [self.pad_id] * (max_len - len(sequence))
        return sequence[:max_len]


@dataclass
class BeamState:
    """Single beam."""

    sequence: List[int]
    log_prob: float
    finished: bool = False

    def add_token(self, token: int, log_prob: float) -> "BeamState":
        """Add a token to this beam."""
        return BeamState(
            sequence=self.sequence + [token],
            log_prob=self.log_prob + log_prob,
            finished=token == 2,  # EOS token
        )


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
        self.concat_latent = concat_latent

        self.token_manager = TokenManager(pad_id, bos_id, eos_id, unk_id)
        self._cached_masks: Dict[int, torch.Tensor] = {}  # Cache for causal mask

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
            num_layers=len(hidden_layers),
        )

        # Output projection
        self.output_layer = nn.Linear(embed_size, vocab_size)

        # Latent connector for conditioning (optional)
        self.latent_connector = (
            nn.Linear(latent_dim, embed_size) if concat_latent else None
        )

    def forward(
        self,
        inputs: Optional[torch.Tensor],
        memory: torch.Tensor,
        mode: str = "TEACH_FORCE",
        gen_type: str = "greedy",
        beam_width: int = 3,
    ) -> torch.Tensor:
        """Route to chosen gen mode."""
        if mode == GenerationMode.TEACH_FORCE.value:
            if inputs is None:
                raise ValueError("Inputs required for teacher forcing mode")
            return self.teacher_force_forward(inputs, memory)
        else:
            return self.generate(memory, gen_type, beam_width)

    def teacher_force_forward(
        self, inputs: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Teacher forcing training pass."""
        memory = self.memory_projection(memory)
        embedded = self._embed_and_condition(inputs, memory)

        causal_mask = self._get_causal_mask(inputs.size(1), inputs.device)
        memory = memory.permute(1, 0, 2)
        dec_out = self._transformer_forward(embedded, memory, causal_mask)

        return self.output_layer(dec_out)

    def generate(
        self, memory: torch.Tensor, gen_type: str = "greedy", beam_width: int = 3
    ) -> torch.Tensor:
        """Route to chosen gen mode."""
        memory = self.memory_projection(memory)

        if gen_type == GenerationMode.GREEDY.value:
            return self.greedy_decode(memory)
        elif gen_type == GenerationMode.BEAM.value:
            return self.beam_decode(memory, beam_width)
        else:
            raise ValueError(f"Unknown generation type: {gen_type}")

    def greedy_decode(self, memory: torch.Tensor) -> torch.Tensor:
        """Greedy decoding with early stopping for inference."""
        batch_size = memory.size(0)
        device = memory.device

        # Initialize with BOS tokens in place of inputs
        current_input = self.token_manager.create_bos_tensor(batch_size, device)
        memory = memory.permute(1, 0, 2)

        outputs = []
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        for step in range(self.max_dec_len):
            if not active_mask.any():
                break

            embedded = self._embed_and_condition(current_input, memory, active_mask)
            causal_mask = self._get_causal_mask(current_input.size(1), device)
            dec_out = self._transformer_forward(
                embedded, memory, causal_mask, active_mask
            )

            logits = self.output_layer(dec_out[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)

            outputs.append(next_token)
            current_input = torch.cat([current_input, next_token], dim=1)

            eos_mask = self.token_manager.is_eos(next_token.squeeze(-1))
            active_mask = active_mask & ~eos_mask

        return torch.cat(outputs, dim=1)

    def beam_decode(self, memory: torch.Tensor, beam_width: int = 3) -> torch.Tensor:
        """Beam search decoding to find best sequences."""
        batch_size = memory.size(0)
        device = memory.device
        memory = memory.permute(1, 0, 2)

        beams = self._initialize_beams(batch_size, beam_width)
        finished_beams = [[] for _ in range(batch_size)]

        for step in range(self.max_dec_len):
            if all(len(beams[b]) == 0 for b in range(batch_size)):
                break

            beams, finished_beams = self._expand_beams_parallel(
                beams, finished_beams, memory, beam_width, device
            )

        return self._select_best_sequences(beams, finished_beams, batch_size, device)

    def _initialize_beams(
        self, batch_size: int, beam_width: int
    ) -> List[List[BeamState]]:
        """Initialize beams for each item in batch."""
        return [
            [BeamState(sequence=[self.token_manager.bos_id], log_prob=0.0)]
            for _ in range(batch_size)
        ]

    def _expand_beams_parallel(
        self,
        beams: List[List[BeamState]],
        finished_beams: List[List[BeamState]],
        memory: torch.Tensor,
        beam_width: int,
        device: torch.device,
    ) -> Tuple[List[List[BeamState]], List[List[BeamState]]]:
        """Expand all beams in parallel."""
        for batch_idx in range(len(beams)):
            if len(beams[batch_idx]) == 0:
                continue

            candidates = []

            for beam in beams[batch_idx]:
                if beam.finished:
                    continue

                seq_tensor = torch.tensor(
                    [beam.sequence], device=device, dtype=torch.long
                )
                memory_batch = memory[:, batch_idx:batch_idx + 1]

                embedded = self._embed_and_condition(seq_tensor, memory_batch)
                causal_mask = self._get_causal_mask(seq_tensor.size(1), device)
                dec_out = self._transformer_forward(embedded, memory_batch, causal_mask)

                logits = self.output_layer(dec_out[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)

                # Get top-k candidates
                top_log_probs, top_indices = log_probs[0].topk(beam_width)

                for i in range(beam_width):
                    new_token = top_indices[i].item()
                    new_log_prob = top_log_probs[i].item()
                    new_beam = beam.add_token(new_token, new_log_prob)
                    candidates.append(new_beam)

            # Select top 'beam_width' candidates
            candidates = sorted(candidates, key=lambda x: x.log_prob, reverse=True)[
                :beam_width
            ]

            new_beams = []
            for beam in candidates:
                if beam.finished:
                    finished_beams[batch_idx].append(beam)
                else:
                    new_beams.append(beam)

            beams[batch_idx] = new_beams

        return beams, finished_beams

    def _select_best_sequences(
        self,
        beams: List[List[BeamState]],
        finished_beams: List[List[BeamState]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Select the best sequences from all beams."""
        outputs = []

        for batch_idx in range(batch_size):
            all_beams = beams[batch_idx] + finished_beams[batch_idx]

            # No valid sequences, return padding
            if len(all_beams) == 0:
                best_seq = [self.token_manager.pad_id] * self.max_dec_len

            # Select beam with highest log probability
            else:

                best_beam = max(all_beams, key=lambda x: x.log_prob)
                best_seq = self.token_manager.pad_sequence(
                    best_beam.sequence, self.max_dec_len
                )

            outputs.append(best_seq)

        return torch.tensor(outputs, device=device, dtype=torch.long)

    def _embed_and_condition(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed inputs and condition with latent if enabled."""
        embedded = self.embedding(inputs)

        if self.concat_latent:
            if memory.dim() == 3:
                latent = self.latent_connector(memory.mean(dim=0))
            else:
                latent = self.latent_connector(memory.mean(dim=1))

            embedded = embedded + latent.unsqueeze(1)

        return embedded

    def _transformer_forward(
        self,
        embedded: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Only process active sequences and stop early where EOS is reached."""
        if active_mask is not None and active_mask.any():
            active_embedded = embedded[active_mask]
            active_memory = memory[:, active_mask]
            
            active_dec_out = self.transformer(
                tgt=active_embedded.permute(1, 0, 2),
                memory=active_memory,
                tgt_mask=causal_mask,
                tgt_is_causal=True,
            ).permute(1, 0, 2)
            
            dec_out = torch.zeros_like(embedded)
            dec_out[active_mask] = active_dec_out

        else:
            dec_out = self.transformer(
                tgt=embedded.permute(1, 0, 2),
                memory=memory,
                tgt_mask=causal_mask,
                tgt_is_causal=True,
            ).permute(1, 0, 2)
            
        return dec_out

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get/create cached causal mask."""
        if seq_len not in self._cached_masks:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            self._cached_masks[seq_len] = mask
        return self._cached_masks[seq_len]
