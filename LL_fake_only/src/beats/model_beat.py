import logging
from pathlib import Path

import torch
import torch.nn as nn

try:
    from src.beats.BEATs import BEATs, BEATsConfig
except ModuleNotFoundError:
    from beats.BEATs import BEATs, BEATsConfig

logger = logging.getLogger(__name__)

ENCODER_DIM = 768   # BEATs encoder hidden size — one layer before the 527 predictor
FEATURE_DIM = 527   # kept for back-compat references; no longer used by model_beat
NUM_HEADS = 8
DEFAULT_SEQ_LEN = 192
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
)


class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, feature_dim=ENCODER_DIM, num_heads=NUM_HEADS, seq_len=DEFAULT_SEQ_LEN):
        super().__init__()
        if seq_len % num_heads != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.chunk_size = seq_len // num_heads
        self.attention_heads = nn.ModuleList(
            [nn.Linear(feature_dim, 1) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [batch, time, feature], got {tuple(x.shape)}")

        batch_size, time_steps, _ = x.shape
        expected_steps = self.num_heads * self.chunk_size
        if time_steps != expected_steps:
            raise ValueError(
                f"Expected {expected_steps} timesteps for attention, got {time_steps}"
            )

        x = x.reshape(batch_size, self.num_heads, self.chunk_size, -1)

        head_outputs = []
        for head_idx, attention_head in enumerate(self.attention_heads):
            chunk = x[:, head_idx, :, :]
            attn_weights = torch.softmax(attention_head(chunk), dim=1)
            attended = (chunk * attn_weights).sum(dim=1)
            head_outputs.append(attended)

        return torch.stack(head_outputs, dim=0).sum(dim=0)


_VALID_FEATURE_LAYERS = ('predictor', 'encoder')


class model_beat(nn.Module):
    def __init__(self, num_label=None, three_loss=False, feature_layer='predictor'):
        """
        Args:
            feature_layer: where to tap BEATs.
                'predictor' (default) → 527-dim AudioSet predictor output.
                'encoder' → 768-dim encoder hidden states (one layer earlier).
        """
        super().__init__()

        if feature_layer not in _VALID_FEATURE_LAYERS:
            raise ValueError(
                f"feature_layer must be one of {_VALID_FEATURE_LAYERS}, got {feature_layer!r}"
            )

        checkpoint = torch.load(str(DEFAULT_CHECKPOINT), weights_only=True)
        cfg = BEATsConfig(checkpoint["cfg"])
        beats_model = BEATs(cfg)
        beats_model.load_state_dict(checkpoint["model"])
        self.BEATs = beats_model

        self.num_label = num_label
        self.three_loss = three_loss
        self.feature_layer = feature_layer
        self.feature_dim = ENCODER_DIM if feature_layer == 'encoder' else FEATURE_DIM

        msg = (f"[model_beat] feature_layer={feature_layer!r} → feature_dim={self.feature_dim} "
               f"(num_label={num_label}, three_loss={three_loss})")
        logger.info(msg)
        print(msg)

        self.temporal_attention = MultiHeadTemporalAttention(
            feature_dim=self.feature_dim, num_heads=NUM_HEADS,
        )

        if not self.three_loss:
            for param in self.BEATs.parameters():
                param.requires_grad = False

        self.last_dropout = nn.Dropout(0.1)
        self.last_layer = nn.Linear(self.feature_dim, self.num_label)
        self.output_softmax = nn.Softmax(dim=1)

        softmax_output_dim = self.num_label if self.num_label is not None else 5
        self.softmax_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, softmax_output_dim),
            nn.Softmax(dim=1),
        )

        self.contrastive_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

    def _extract_encoder_features(self, source: torch.Tensor) -> torch.Tensor:
        """Tap BEATs at the encoder output (768-dim) — one layer before the 527 predictor."""
        beats = self.BEATs
        fbank = beats.preprocess(source)
        fbank = fbank.unsqueeze(1)
        features = beats.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = beats.layer_norm(features)
        if beats.post_extract_proj is not None:
            features = beats.post_extract_proj(features)
        x = beats.dropout_input(features)
        x, _ = beats.encoder(x, padding_mask=None)
        return x  # [B, T, 768]

    def forward_pipeline(self, i_tensor: torch.Tensor):
        if self.feature_layer == 'encoder':
            x = self._extract_encoder_features(i_tensor)        # [B, T, 768]
        else:
            x = self.BEATs.extract_features(i_tensor)[0]        # [B, T, 527]

        x = self.temporal_attention(x)
        x = self.last_dropout(x)

        if self.three_loss:
            bonafide_head = x
            softmax_head = self.softmax_head(x)
            contrastive_head = self.contrastive_head(x)
            return bonafide_head, softmax_head, contrastive_head

        logits = self.last_layer(x)
        return self.output_softmax(logits)
