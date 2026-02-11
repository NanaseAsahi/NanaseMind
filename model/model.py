import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Union
import math

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

def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    """
    复制kv的每个头 使得K,V的头数与Q的头数相同
    n_rep: 复制的次数

    等效torch.repeat_interleave(x, dim=2, repeats=n_rep) 
    但repeat_interleave会复制内存 开销大很多
    而下面的实现只是在view时改变了shape
    x: [B, L, n_head, head_dim]
    """
    if n_rep == 1:
        return x
    
    B, L, H, D = x.shape

    x = x[:, :, :, None, :].expand(B, L, H, n_rep, D).reshape(B, L, H*n_rep, D)
    return x

class Attention(nn.Module):
    def __init__(self, args: NanaseMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0, \
        "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads  # 当前类中要使用的n_head
        self.n_local_kv_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // self.n_local_heads

        self.q_proj = nn.Linear(args.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention
    
    def forward(self, 
                x:torch.Tensor, 
                pos_emb: Tuple[torch.Tensor, torch.Tensor],  # cos sin
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor]=None):
        
        B, L, _ = x.shape

        # [B, L, n_head*head_dim]  [B, L, n_kv_head*head_dim]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x) 

        xq = xq.view(B, L, self.n_local_heads, self.head_dim)
        xk = xk.view(B, L, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(B, L, self.n_local_kv_heads, self.head_dim)

        cos, sin = pos_emb
        xq, xk = apply_rotary_pos_emd(xq, xk, cos[:L], sin[:L])

        # kv cache
        if past_kv is not None:
            # 注意kv cache只在推理时启用 此时token是一个一个输入进来算的
            # 故在dim=1上拼接
            xk = torch.cat([past_kv[0], xk], dim=1)
            xv = torch.cat([past_kv[1], xv], dim=1)
        
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),  # [B, L, n_head, dim] -> [B, n_head, L, dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 1. torch版本是否符合要求且是否人为要求使用Flash Attention
        # 2. seq_len是否大于1 因为seq_len=1（也即推理时）时Flash Attention的优势不明显 反而可能更慢
        # 3. 是否使用kv cache 理由类似2 使用kv cache时即推理时
        # 4. Flash Attention 默认causal 不支持 mask （Flash Attention自己实现了"不看未来"的mask）
        if self.flash_attention and (L > 1) and (past_kv is None) and (attention_mask is None or torch.all(attention_mask==1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, n_head, n_q, n_k]

            # causal mask 不看未来
            # 在训练时scores[B, n_head, n_q, n_k] n_q=n_k=L 实际上就是对scores[:, :, :, :]加上一个mask
            # 但在推理时 由于有kv cache（假设长度为T）的存在，那么应该是对T之后的scores加上mask 也就是[:, :, :. -L:]
            scores[:, :, :, -L:] += torch.triu(torch.full((L, L), float('-inf'), device=scores.device), diagonal=1)

            # padding mask 不看padding
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores, dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores.matmul(xv)

        # [B, n_head, L, head_dim] -> [B, L, n_head*head_dim]
        output = output.transpose(1, 2).reshape(B, L, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv
    
class FeedForward(nn.Module):
    def __init__(self, args: NanaseMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            # 针对SwiGLU GeGLU 让二者的参数量与使用ReLU时相近
            intermediate_size = int(args.hidden_size * 8 / 3)
            # 向上取整到64的倍数
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        # ACT2FN是一个dict 将字符串映射到激活函数(torch实现)
        self.act_fn = ACT2FN[args.hidden_act]
    
    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
    
class MOEFeedForward(nn.Module):
    def __init__(self, args: NanaseMindConfig):
        super().__init__()
    
    def forward(self, x):
        pass
    
class NanaseMindBlock(nn.Module):
    def __init__(self, layer_id: int, args: NanaseMindConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = args.hidden_size // self.num_attention_heads
        self.attn = Attention(args)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(self.hidden_size, eps=args.rms_norm_eps)
        self.post_layernorm = RMSNorm(self.hidden_size, eps=args.rms_norm_eps)
        self.mlp = FeedForward(args) if not args.use_moe else MOEFeedForward(args)
    
    def forward(self, hidden_states, position_embeddings, past_kv=None, use_cache=False, attention_mask=None):
        # 存储一个Block最初的输入
        residual = hidden_states
        hidden_states, present_kv = self.attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_kv,
            use_cache,
            attention_mask
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_layernorm(hidden_states))

        return hidden_states, present_kv

class NanaseMindModel(nn.Module):
    def __init__(self, args: NanaseMindConfig):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList([NanaseMindBlock(i, args) for i in range(self.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_feqs_cis(
            dim=args.hidden_size // args.num_attention_heads,
            end=args.max_position_embeddings,
            rope_base=args.rope_theta,
            repe_scaling=args.rope_scaling
        )

        # 不属于模型参数 但需要跟着模型移动 自动.to(device) 自动进入state_dict(persistent参数决定)
        self.register_buffer("freq_cos", freqs_cos, persistent=False)
        self.register_buffer("freq_sin", freqs_sin, persistent=False)
    
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,  # token -> token id
                attention_mask: Optional[torch.Tensor] = None,
                past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None,
                use_cache: bool = False,
                **kwargs):
        B, L = input_ids.shape

        # if transformers style -> past_kv = None 兼容transformers的输入格式
        if hasattr(past_kvs, 'layers'): past_kvs = None
        past_kvs = past_kvs or [None] * len(self.layers)
        start_pos = past_kvs[0][0].shape[1] if past_kvs[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + L],
            self.freqs_sin[start_pos: start_pos + L]
        )

        presents = [] 
        for layer_idx, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_kv=past_kv,
                use_cache=use_cache,
                attention_mask=attention_mask
            )

            presents.append(present)

            hidden_states = self.norm(hidden_states)

        # hidden_states.new_zeros(1).squeeze()创建一个与hidden_states同设备同dtype的标量0 当MoE为空/0时返回0作为aux_loss
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], 
                        hidden_states.new_zeros(1).squeeze())
            
        return hidden_states, presents, aux_loss
    
class NanaseMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanaseMindConfig

    def __init__(self, config: NanaseMindConfig = None):
        self.config = config or NanaseMindConfig()
        super().__init__()
        self.model = NanaseMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # 权重共享
    
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs):
        
        hidden_states, past_kv, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache,
            **kwargs
        )

        # slice 封装一个切片规则 类似一个切片中传入一个可变的参数 slice(start, end, step) 从start到end-1 每step个取一个
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # hidden_states: [:, -logits_to_keep:, :]

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_kv, hidden_states=hidden_states)
        output.aux_loss = aux_loss

        return output

if __name__ == "__main__":
    """RoPE & YaRN Test"""
    # batch_size = 2
    # seq_len = 512
    # num_heads = 8
    # head_dim = 64  # hidden_size // num_heads = 512 // 8 = 64

    # q = torch.randn(batch_size, seq_len, num_heads, head_dim)  # [2, 512, 8, 64]
    # k = torch.randn(batch_size, seq_len, num_heads, head_dim)  # [2, 512, 8, 64]
    
    # print("=" * 50)
    # rope_scaling = {
    #     "original_max_position_embeddings": 2048,
    #     "factor": 4,
    #     "beta_fast": 4,
    #     "beta_slow": 1,
    #     "type": "yarn",
    # }
    # cos_yarn, sin_yarn = precompute_feqs_cis(dim=head_dim, end=seq_len, rope_scaling=rope_scaling)
    # print(f"cos_yarn shape: {cos_yarn.shape}")  # [seq_len, head_dim]
    # print(f"sin_yarn shape: {sin_yarn.shape}")  # [seq_len, head_dim]
    
    # q_embed_yarn, k_embed_yarn = apply_rotary_pos_emd(q, k, cos_yarn, sin_yarn, unsqueeze_dim=1)
    # print(f"q_embed_yarn shape: {q_embed_yarn.shape}")  # [2, 512, 8, 64]
    # print(f"k_embed_yarn shape: {k_embed_yarn.shape}")  # [2, 512, 8, 64]


