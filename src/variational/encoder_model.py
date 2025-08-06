import lightning as L
import torch
import torch.nn as nn

from src.utils import get_activation


class EncoderModel(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        latent_dim: int,
        hidden_layers: list = [128, 128],
        nhead: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
        pad_id: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.latent_dim = latent_dim
        self.nhead = nhead
        self.dropout = dropout
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.pad_id = pad_id

        self.transformer_layers = nn.ModuleList()
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):

            # Projection layer
            if i == 0:
                self.transformer_layers.append(nn.Linear(prev_dim, hidden_dim))
                self.transformer_layers.append(get_activation(activation))
                prev_dim = hidden_dim

            # Transformer encoder layer
            else:
                self.transformer_layers.append(
                    nn.TransformerEncoderLayer(
                        d_model=prev_dim,
                        nhead=nhead,
                        dim_feedforward=hidden_dim,
                        dropout=dropout,
                        activation=activation,
                    )
                )
                # Additional projection
                self.transformer_layers.append(nn.Linear(prev_dim, hidden_dim))
                self.transformer_layers.append(get_activation(activation))
                prev_dim = hidden_dim

        # Output projection
        self.output_layer = nn.Linear(prev_dim, memory_dim)

        # Q(z | x) networks
        self.mu = nn.Linear(memory_dim, latent_dim)
        self.logvar = nn.Linear(memory_dim, latent_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input tokens to latent space.

        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            mask: Padding mask (batch_size, seq_len), True for padding tokens

        Returns:
            z_mu: Mean of latent distribution (batch_size, latent_dim)
            z_logvar: Log variance of latent distribution (batch_size, latent_dim)
            hidden_st: Hidden states (batch_size, seq_len, memory_dim)
        """
        hidden = x

        # Transformer encoder layers
        for i, layer in enumerate(self.transformer_layers):
            if isinstance(layer, nn.TransformerEncoderLayer):
                hidden = hidden.permute(1, 0, 2)
                hidden = layer(hidden, src_key_padding_mask=mask)
                hidden = hidden.permute(1, 0, 2)
            else:
                hidden = layer(hidden)

        hidden_st = self.output_layer(hidden)

        # Pooling for sequence-level latent variables
        if mask is not None:
            valid_mask = (~mask).float().unsqueeze(-1)
            masked_hidden = hidden_st * valid_mask
            sum_hidden = masked_hidden.sum(dim=1)
            valid_counts = valid_mask.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / valid_counts
        else:
            pooled = hidden_st.mean(dim=1)

        z_mu = self.mu(pooled)
        z_logvar = self.logvar(pooled)

        return z_mu, z_logvar, hidden_st
