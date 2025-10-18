import torch
from ml.src.diffusion.hooks import CrossFrameAttnProcessor, SharedAttentionContext, SharedAttentionConfig

class DummyAttention:
    def __init__(self, heads=8, head_dim=64, seq_len=10, kv_len=64):
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = torch.nn.Linear(128, heads * head_dim)
        self.to_k = torch.nn.Linear(128, heads * head_dim)
        self.to_v = torch.nn.Linear(128, heads * head_dim)
        self.spatial_norm = None
        self.group_norm = None
        self.norm_cross = None
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(heads * head_dim, 128)])
        self.residual_connection = True

    def get_attention_scores(self, q, k, attention_mask=None):
        # q: (B, H, S, D), k: (B, H, T, D)
        return torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

def test_crossframe_shapes():
    cfg = SharedAttentionConfig()
    ctx = SharedAttentionContext(cfg=cfg, batch_size=2)
    proc = CrossFrameAttnProcessor(ctx)

    attn = DummyAttention()
    hidden_states = torch.randn(2, 10, 128)
    encoder_hidden_states = torch.randn(2, 64, 128)

    residual, input_ndim, B, C, H, W, q, k, v = proc._project_qkv(
        attn, hidden_states, encoder_hidden_states, None, None
    )

    print("q shape:", q.shape)
    print("k shape:", k.shape)
    print("v shape:", v.shape)
    assert q.shape[2] == k.shape[2] == v.shape[2], "Sequence lengths must match"
    print("✅ Test passed: q/k/v shapes aligned correctly")

if __name__ == "__main__":
    test_crossframe_shapes()
