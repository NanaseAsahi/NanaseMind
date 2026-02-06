import torch
import torch.nn as nn
from model import NanaseMindConfig, Attention, precompute_feqs_cis
import math

def test_attention_module():
    """测试Attention模块的功能"""
    
    # 配置参数
    config = NanaseMindConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA配置
        dropout=0.1,
        flash_attention=False,  # 为了简化测试，先关闭Flash Attention
    )
    
    # 创建Attention模块
    attention = Attention(config)
    attention.eval()  # 评估模式
    
    # 测试参数
    batch_size = 2
    seq_len = 16
    hidden_size = config.hidden_size
    head_dim = config.hidden_size // config.num_attention_heads
    
    print("=" * 60)
    print("Attention Module Test")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Number of KV heads: {config.num_key_value_heads}")
    print(f"Head dim: {head_dim}")
    print(f"Head multiplication factor (n_rep): {config.num_attention_heads // config.num_key_value_heads}")
    print()
    
    # 1. 测试基本的前向传播
    print("Test 1: Basic Forward Pass")
    print("-" * 60)
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # 预计算RoPE位置编码
    cos, sin = precompute_feqs_cis(dim=head_dim, end=seq_len)
    pos_emb = (cos.to(x.device), sin.to(x.device))
    
    try:
        output, past_kv = attention(x, pos_emb, use_cache=False)
        print(f"✓ Forward pass succeeded")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  past_kv: {past_kv}")
        print()
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        print()
        return
    
    # 2. 测试输出形状是否正确
    print("Test 2: Output Shape Verification")
    print("-" * 60)
    
    expected_shape = (batch_size, seq_len, hidden_size)
    if output.shape == expected_shape:
        print(f"✓ Output shape is correct: {output.shape}")
    else:
        print(f"✗ Output shape mismatch!")
        print(f"  Expected: {expected_shape}")
        print(f"  Got: {output.shape}")
    print()
    
    # 3. 测试KV缓存（使用cache）
    print("Test 3: KV Cache Test")
    print("-" * 60)
    
    try:
        # 第一次调用，使用cache
        x_single = torch.randn(batch_size, 1, hidden_size)
        cos_single, sin_single = precompute_feqs_cis(dim=head_dim, end=seq_len)
        pos_emb_single = (cos_single.to(x_single.device), sin_single.to(x_single.device))
        
        output1, past_kv = attention(x_single, pos_emb_single, use_cache=True)
        print(f"✓ First call (use_cache=True) succeeded")
        print(f"  Output shape: {output1.shape}")
        if past_kv is not None:
            print(f"  past_kv[0] (K cache) shape: {past_kv[0].shape}")
            print(f"  past_kv[1] (V cache) shape: {past_kv[1].shape}")
        print()
        
        # 第二次调用，使用之前缓存的KV
        x_next = torch.randn(batch_size, 1, hidden_size)
        output2, past_kv = attention(x_next, pos_emb_single, past_kv=past_kv, use_cache=True)
        print(f"✓ Second call (with past_kv) succeeded")
        print(f"  Output shape: {output2.shape}")
        if past_kv is not None:
            print(f"  Updated past_kv[0] (K cache) shape: {past_kv[0].shape}")
            print(f"  Updated past_kv[1] (V cache) shape: {past_kv[1].shape}")
        print()
    except Exception as e:
        print(f"✗ KV cache test failed: {e}")
        print()
    
    # 4. 测试attention mask
    print("Test 4: Attention Mask Test")
    print("-" * 60)
    
    try:
        # 创建一个attention mask（例如，填充mask）
        # 1表示有效位置，0表示填充位置
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 10:] = 0  # 后8个token作为填充
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output, past_kv = attention(x, pos_emb, attention_mask=attention_mask, use_cache=False)
        print(f"✓ Forward pass with attention mask succeeded")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Output shape: {output.shape}")
        print()
    except Exception as e:
        print(f"✗ Attention mask test failed: {e}")
        print()
    
    # 5. 测试梯度流
    print("Test 5: Gradient Flow Test")
    print("-" * 60)
    
    attention.train()  # 训练模式
    
    try:
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        output, _ = attention(x, pos_emb, use_cache=False)
        
        # 计算一个标量loss
        loss = output.sum()
        loss.backward()
        
        if x.grad is not None:
            print(f"✓ Gradient flow successful")
            print(f"  Input gradient shape: {x.grad.shape}")
            print(f"  Input gradient norm: {x.grad.norm().item():.6f}")
        
        # 检查模块参数是否有梯度
        has_grad = False
        for name, param in attention.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if has_grad:
            print(f"✓ Module parameters have gradients")
        else:
            print(f"✗ Module parameters have no gradients!")
        print()
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        print()
    
    # 6. 测试数值稳定性
    print("Test 6: Numerical Stability Test")
    print("-" * 60)
    
    attention.eval()
    
    try:
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # 检查是否有NaN或Inf
        with torch.no_grad():
            output, _ = attention(x, pos_emb, use_cache=False)
        
        if torch.isnan(output).any():
            print(f"✗ Output contains NaN values!")
        elif torch.isinf(output).any():
            print(f"✗ Output contains Inf values!")
        else:
            print(f"✓ Output is numerically stable (no NaN or Inf)")
            print(f"  Output mean: {output.mean().item():.6f}")
            print(f"  Output std: {output.std().item():.6f}")
        print()
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")
        print()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_attention_module()
