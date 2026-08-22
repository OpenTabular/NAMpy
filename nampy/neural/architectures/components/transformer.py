import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        assert x.size(-1) % 2 == 0, "Input dimension must be even"
        split_dim = x.size(-1) // 2
        return x[..., :split_dim] * torch.sigmoid(x[..., split_dim:])


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, activation=F.relu, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(
            *args, activation=activation, **kwargs
        )
        self.custom_activation = activation

        # Check if the activation function is an instance of a GLU variant
        if activation is GLU or isinstance(activation, GLU):
            self.linear1 = nn.Linear(
                self.linear1.in_features,
                self.linear1.out_features * 2,
                bias=kwargs.get("bias", True),
            )
            self.linear2 = nn.Linear(
                self.linear2.in_features,
                self.linear2.out_features,
                bias=kwargs.get("bias", True),
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Keep the same pre/post-norm control flow as
        # torch.nn.TransformerEncoderLayer, including causal attention.  The
        # only intentional variation is the configurable activation (including
        # GLU, whose doubled linear1 width is set in __init__).
        if self.norm_first:
            src = src + self._sa_block(
                self.norm1(src), src_mask, src_key_padding_mask, is_causal
            )
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(
                src + self._sa_block(src, src_mask, src_key_padding_mask, is_causal)
            )
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.custom_activation(self.linear1(x))
        dropout_p = self.dropout.p if isinstance(self.dropout, nn.Dropout) else self.dropout
        x = F.dropout(x, p=dropout_p, training=self.training)
        return self.dropout2(self.linear2(x))
