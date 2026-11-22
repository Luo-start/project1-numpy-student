"""Pure-NumPy image encoder for the fixed OpenCLIP ViT-B/32 checkpoint.

The implementation covers the six assessed forward-pass components:
patch extraction, patch/CLS embedding, attention-head partitioning,
multi-head attention, Transformer-block evaluation, and the final CLS head.
All model parameters are supplied; no training or PyTorch dependency is used.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    """Evaluate scaled multi-head self-attention over token rows.

    Stored linear weights use the ``[out_features, in_features]`` layout, so
    every learned linear transformation is applied as ``x @ W.T + b``.
    """

    def __init__(self, weights, prefix, heads):
        # The input projection stores concatenated Q, K and V parameters.
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """Move the feature axis into independent attention heads.

        Args:
            x: Token features with shape ``[B,N,D]``.

        Returns:
            Head-separated features with shape ``[B,H,N,D/H]``. The token
            order is unchanged; only the feature axis is partitioned.

        For this checkpoint, ``D=768``, ``H=12`` and ``D/H=64``.
        """
        B,N,D=x.shape          # batch, token count, hidden width
        H=self.heads           # 12 attention heads
        D_h=self.head_dim      # 64 features per head
        x=x.reshape(B,N,H,D_h)  # [B,N,H,D/H]
        x=x.transpose(0,2,1,3)  # [B,H,N,D/H]
        return x

    def __call__(self, x, causal=False):
        """Apply multi-head attention to ``[B,N,D]`` token rows.

        The optional causal mask is retained for compatibility with the later
        text encoder. The image encoder always evaluates attention without it.
        """
        # Fuse the three input projections, then split the last axis into Q/K/V.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # Dot products between every query/key pair produce [B,H,N,N] scores.
        # Keep the scale as float32 so NumPy does not promote the scores.
        scale=np.float32(np.sqrt(self.head_dim))   # sqrt(64)=8.0, dtype=float32
        k_transpose=k.transpose(0,1,3,2)           # [B,H,64,N]
        scores=(q@k_transpose)/scale                # [B,H,N,N]
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)    # row-wise attention weights

        # Mix value rows, then restore the original token/head axis order.
        context=coefficients @ v                    # [B,H,N,D/H]
        B, H, N, D_h=context.shape
        merged=context.transpose(0,2,1,3).reshape(B,N,H*D_h)  # [B,N,D]

        # Project the concatenated heads back to the model width.
        output=merged@self.out_weight.T+self.out_bias  # [B,N,D]
        return output

class TransformerBlock:
    """Evaluate one fixed-weight Pre-LN Transformer block."""

    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        """Apply attention and MLP residual paths while preserving [B,N,D].

        The block follows ``x + MHA(LN(x))`` followed by
        ``residual + MLP(LN(residual))``. All parameters are fixed.
        """
        # First residual branch: normalized attention update plus the input.
        update = self.attention(self.norm1(x), causal=causal)
        residual = x + update  # [B,N,D]

        # Second residual branch: expand to 4D, apply GELU, then project to D.
        u = self.norm2(residual)
        hidden=u@self.fc1_weight.T+self.fc1_bias    # [B,N,3072]
        hidden=self.gelu(hidden)                     # exact erf-based GELU
        mlp_update=hidden@self.fc2_weight.T+self.fc2_bias  # [B,N,768]
        return residual+mlp_update

class ImageEncoder:
    """Complete ViT-B/32 image tower from pixels to normalized features."""

    def __init__(self, weights):
        # The convolution kernel is reused as a [hidden, patch_values] matrix.
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)  # [1,1,D]
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """Convert images into flattened 32x32 RGB patches.

        Args:
            images: Preprocessed images with shape ``[B,3,224,224]``.

        Returns:
            Patch rows with shape ``[B,49,3072]``. Patches are ordered left to
            right and then top to bottom; values inside each patch follow the
            channel/height/width (C,H,W) order.
        """
        B=images.shape[0]                    # keep the batch axis independent
        x=images.reshape(B,3,7,32,7,32)     # split height/width into patch grids
        x=x.transpose(0,2,4,1,3,5)          # [B,patch_h,patch_w,C,32,32]
        x=x.reshape(B,49,3072)              # flatten 3*32*32 values per patch
        return x

    def embed(self, images):
        """Project patches and build the CLS-plus-patch token sequence.

        Args:
            images: Preprocessed images with shape ``[B,3,224,224]``.

        Returns:
            Token embeddings with shape ``[B,50,768]`` after adding the learned
            positional embedding to one CLS row and 49 patch rows.
        """
        patches = self.patchify(images)  # [B,49,3072]

        # The checkpoint's patch projection has no bias.
        patch_tokens=patches@self.patch_weight.T  # [B,49,768]

        # Repeat the learned CLS row once for every image in the batch.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # CLS is token zero; broadcasts the shared [50,768] positions over B.
        tokens=np.concatenate([cls,patch_tokens],axis=1)  # [B,50,768]
        return tokens+self.position

    def encode_tokens(self, images):
        """Run pre-Norm and all twelve Transformer blocks.

        Input and output both have shape ``[B,50,768]``.
        """
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """Return one L2-normalized 512-dimensional feature per image.

        The CLS token is selected after the twelve blocks, post-normalized,
        projected from 768 to 512 dimensions, and normalized row by row.
        """
        tokens = self.encode_tokens(images)

        # Token zero is the CLS representation: [B,50,768] -> [B,768].
        cls=tokens[:,0,:]
        cls=self.post_norm(cls)  # [B,768]

        # visual.proj is already [in,out], so no transpose is needed here.
        output=cls@self.projection  # [B,512]
        return unit_vector(output)
