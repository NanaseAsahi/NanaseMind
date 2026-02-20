import torch
import torch.nn as nn
import torch.nn.init as init
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
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
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
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # x: [B, n_heads, seq_len, d_hidden]
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_feqs_cis(dim: int, end: int = 32*1024, rope_base: float = 1e6,
                        rope_scaling: Optional[dict] = None):
    """
    Precompute the freqs and cis used for RoPE(correction based YaRN).

    dim: d_hidden // num_head if num_head > 1 else d_hidden
    end: max seq_len
    rope_base: the base value for w_k (1000000)
    rope_scaling: dict or None, for YaRN scaling
    """
    # 标准RoPE theta_i = 1 / base ^ (2i/d)
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )

        if end / orig_max > 1.0:
            # 当当前的最大长度大于原本训练时的长度时才缩放
            # YaRN: f'(i) = f(i)((1-gamma) + gamma/s) where gamma \in [0, 1] is linear ramp

            # 传入b（转了多少圈） 求出在哪个维度上转了b圈 
            inv_dim = lambda b : (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 求得低频、高频的分界维度
            low, high = max(math.floor(inv_dim(beta_fast), 0)), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1-ramp + ramp / factor)
    
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin

def apply_rotary_pos_emd(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def _DEPRECATED_precompute_feqs_cis(dim: int, end: int=32*1024, rope_base:float=1e6, rope_scaling:Optional[dict]=None):
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

def _DEPRECATED_apply_rotary_pos_emd(q, k, cos, sin, unsqueeze_dim: int=1):
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
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
    
    def forward(self, 
                x:torch.Tensor, 
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # cos sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor]=None):
        
        B, L, _ = x.shape

        # [B, L, n_head*head_dim]  [B, L, n_kv_head*head_dim]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x) 

        xq = xq.view(B, L, self.n_local_heads, self.head_dim)
        xk = xk.view(B, L, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(B, L, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emd(xq, xk, cos, sin)

        # kv cache
        if past_key_value is not None:
            # 注意kv cache只在推理时启用 此时token是一个一个输入进来算的
            # 故在dim=1上拼接
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
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
        if self.flash and (L > 1) and (past_kv is None) and (attention_mask is None or torch.all(attention_mask==1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, n_head, n_q, n_k]

            # causal mask 不看未来
            # 训练时: scores[B, n_head, L, L], 对所有位置应用上三角mask
            # 推理时: scores[B, n_head, L, T+L], 新输入的L个token可以看到所有cached的T个token
            #        但新token之间需要causal mask, 因此只对最后L列[:, :, :, -L:]应用上三角mask
            scores[:, :, :, -L:] += torch.triu(torch.full((L, L), float('-inf'), device=scores.device), diagonal=1)

            # padding mask 不看padding
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
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
class MoEGate(nn.Module):
    def __init__(self, args: NanaseMindConfig):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        self.scoring_func = args.scoring_func
        self.alpha = args.aux_loss_alpha  # aux_loss在总loss中的权重
        self.seq_aux = args.seq_aux  # 使用序列级别还是使用批次级别

        self.norm_topk_prob = args.norm_topk_prob
        self.gating_dim = args.hidden_size
        # 分配好空间 但不初始化值
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        # hidden_states: [B, L, d_hidden]
        B, L, H = hidden_states.shape
        hidden_states = hidden_states.view(-1, H)  # [B*L, d_hidden]
        # [B*L, d_hidden] @ [n_routed_experts, gating_dim] bias=None -> [B*L, n_routed_experts]
        # [B, in_dim] @ [out_dim, in_dim]
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 权重归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        # 计算aux_loss
        # 只有在训练阶段且alpha>0时才计算aux_loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(B, -1)

            # 序列级别的aux_loss 单独取每一个样本计算
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(B, L, -1)
                # ce用来计算每个样本中 每个专家被选取的次数
                ce = torch.zeros(B, self.n_routed_experts, device=hidden_states.device)
                # 将选取的次数加进ce 最后除以每个专家被选取的平均次数 求得每个专家的负载
                ce.scatter_add_(1, topk_idx_for_aux_loss, 
                                torch.ones(B, L * aux_topk, device=hidden_states.device)).div_(
                                    L * aux_topk / self.n_routed_experts
                                )
                # ce: [B, n_routed_experts] scores_for_seq_aux: [B, L, n_routed_experts] -> [B, n_routed_experts]
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            # 批次级别的aux_loss
            else:
                # 从batch角度计算专家被选取的次数
                # mask_ce: [B * L * aux_topk, n_routed_experts]
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # [n_routed_experts] 每个专家被选取的平均次数
                ce = mask_ce.float().mean(0)
                # [n_routed_experts]
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    def __init__(self, config: NanaseMindConfig):
        super().__init__()
        self.config = config

        # 专家层
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])

        # 门控层
        self.gate = MoEGate(config)

        # 共享专家层 每个token都会经过这些层
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])
        
    def forward(self, x):
        # x: [bsz, seq_len, hidden_size]
        identity = x  # 保存原始输入，用于共享专家的残差连接
        orig_shape = x.shape  # 保存原始形状 [bsz, seq_len, hidden_size]
        B, L, _ = x.shape
        
        # 使用门控机制选择专家
        # topk_idx: [bsz*seq_len, num_experts_per_tok] 每个token选择的top-k个专家索引
        # topk_weight: [bsz*seq_len, num_experts_per_tok] 对应的专家权重
        # aux_loss: 负载均衡的辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 展平：[bsz, seq_len, hidden_size] -> [bsz*seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])

        # 展平索引：[bsz*seq_len, num_experts_per_tok] -> [bsz*seq_len*num_experts_per_tok]
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 训练模式：为每个token的每个选中的专家创建副本
            # [bsz*seq_len, hidden_size] -> [bsz*seq_len*num_experts_per_tok, hidden_size]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)

            # 一个空向量 用来存储专家的输出
            y = torch.empty_like(x, dtype=x.dtype)  # [bsz*seq_len*num_experts_per_tok, hidden_size]
            
            # 遍历每个专家，处理分配给它的tokens
            for i, expert in enumerate(self.experts):
                # flat_topk_idx == i 找出分配给当前专家的所有token副本
                expert_out = expert(x[flat_topk_idx == i])  # [num_tokens_for_expert_i, hidden_size]
                if expert_out.shape[0] > 0: 
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: 
                    # 防止未使用的专家梯度消失：添加一个不影响输出的小项
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            
            # 重塑并应用专家权重：[bsz*seq_len*num_experts_per_tok, hidden_size] 
            # -> [bsz*seq_len, num_experts_per_tok, hidden_size]
            # 乘以权重 [bsz*seq_len, num_experts_per_tok, 1]
            # sum(dim=1) -> [bsz*seq_len, hidden_size]
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 恢复原始形状：[bsz*seq_len, hidden_size] -> [bsz, seq_len, hidden_size]
            y = y.view(*orig_shape)
        else:
            # 推理模式：调用优化的推理函数
            # x: [bsz*seq_len, hidden_size]
            # flat_topk_idx: [bsz*seq_len*num_experts_per_tok]
            # topk_weight: [bsz*seq_len*num_experts_per_tok, 1]
            # 输出: [bsz*seq_len, hidden_size] -> view -> [bsz, seq_len, hidden_size]
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 如果有共享专家，将其输出加到路由专家的输出上（残差连接）
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)  # identity: [bsz, seq_len, hidden_size]
        
        self.aux_loss = aux_loss  # 保存aux_loss用于后续加到总损失中
        return y  # [bsz, seq_len, hidden_size]    

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

    
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
    
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 存储一个Block最初的输入
        residual = hidden_states
        hidden_states, present_key_value = self.attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_layernorm(hidden_states))

        return hidden_states, present_key_value

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
            rope_scaling=args.rope_scaling
        )

        # 不属于模型参数 但需要跟着模型移动 自动.to(device) 自动进入state_dict(persistent参数决定)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,  # token -> token id
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None,
                use_cache: bool = False,
                **kwargs):
        B, L = input_ids.shape

        # 兼容transformers
        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + L],
            self.freqs_sin[start_pos: start_pos + L]
        )

        presents = [] 
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )

            presents.append(present)

        # 最终的 LayerNorm 应该在所有 layer 之后只做一次
        hidden_states = self.norm(hidden_states)

        # hidden_states.new_zeros(1).squeeze()创建一个与hidden_states同设备同dtype的标量0 当MoE为空/0时返回0作为aux_loss
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], 
                        hidden_states.new_zeros(1).squeeze())
            
        return hidden_states, presents, aux_loss
    
class NanaseMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanaseMindConfig

    def __init__(self, config: NanaseMindConfig = None):
        self.config = config or NanaseMindConfig()
        super().__init__(self.config)
        self.model = NanaseMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享
        self.model.embed_tokens.weight = self.lm_head.weight
    
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs):
        
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        # slice 封装一个切片规则 类似一个切片中传入一个可变的参数 slice(start, end, step) 从start到end-1 每step个取一个
        # 训练时logits_to_keep=0 则计算所有位置的logits 用于计算loss
        # 推理时logits_to_keep=1 则取最后一个logits 因为一次只预测一个token
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # hidden_states: [:, -logits_to_keep:, :]

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
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


