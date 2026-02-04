from transformers import PretrainedConfig
import torch
import torch.nn as nn
from typing import Optional

class NanaseMindConfig(PretrainedConfig):
    model_type = "nanasemind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float=1e-6):
        # dim为输入的最后一维的维度
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # [B, L, d_hidden]
        x *= torch.rsqrt(pow(x, 2).mean(dim=-1, keepdim=True) + self.eps)  # rsqrt = 1/sqrt(...)

        return x

    def forward(self, x):
        # type_as == (..., dtype = xxx, device=xxx) 
        return self.weight * self._norm(x).type_as(x)
    
def precompute_feqs_cis(dim: int, end: int=32*1024, rope_base:float=1e6, rope_scaling:Optional[dict]=None):
    """
    Precompute the freqs and cis used for RoPE(correction based YaRN).

    dim: d_hidden // num_head if num_head > 1 else d_hidden
    end: max seq_len
    rope_base: the base value for w_k (1000000)
    rope_scaling: dict or None, for YaRN scaling
    """

    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:  # check self.rope_scaling
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings"),
            rope_scaling.get("factor"),
            rope_scaling.get("beta_fast"),
            rope_scaling.get("beta_slow"),
        )

        # 2*pi / freq -> 波长(seq_len维度上一个周期的长度)
        # 找到第一个波长大于orig_max的位置 作为YaRN高低频处理的分界点 若没有满足的位置则取默认值dim//2
        corr_dim = next((i for i in range(dim//2) if 2*torch.pi/freqs[i] > orig_max), dim // 2)

        power = torch.arange(0, dim // 2) / max(dim // 2 - 1, 1)

        beta = beta_slow + (beta_fast - beta_slow) * power

        scale = torch.where(  # 分段函数的实现
            torch.arange(dim//2, device=freqs.device) < corr_dim,  # condition
            (beta*factor-beta+1) / (beta*factor),  # for high freq & short seq
            1.0/factor  # for low freq & long seq
        ) 

        freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    
    # 也有实现使用repeat_interleave
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

    return freqs_cos, freqs_sin

def apply_rotary_pos_emd(q, k, cos, sin, unsqueeze_dim: int=1):
    def rotate_half(x):
        # 将后半部分旋转到前半部分并取-号
        return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)

    q_embed = (q*cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_embed = (k*cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k)*sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed
    

if __name__ == "__main__":
    """RoPE & YaRN Test"""
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64  # hidden_size // num_heads = 512 // 8 = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)  # [2, 512, 8, 64]
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)  # [2, 512, 8, 64]
    
    print("=" * 50)
    rope_scaling = {
        "original_max_position_embeddings": 2048,
        "factor": 4,
        "beta_fast": 4,
        "beta_slow": 1,
        "type": "yarn",
    }
    cos_yarn, sin_yarn = precompute_feqs_cis(dim=head_dim, end=seq_len, rope_scaling=rope_scaling)
    print(f"cos_yarn shape: {cos_yarn.shape}")  # [seq_len, head_dim]
    print(f"sin_yarn shape: {sin_yarn.shape}")  # [seq_len, head_dim]
    
    q_embed_yarn, k_embed_yarn = apply_rotary_pos_emd(q, k, cos_yarn, sin_yarn, unsqueeze_dim=1)
    print(f"q_embed_yarn shape: {q_embed_yarn.shape}")  # [2, 512, 8, 64]
    print(f"k_embed_yarn shape: {k_embed_yarn.shape}")  # [2, 512, 8, 64]


