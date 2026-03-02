import torch
from torch import optim, nn

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()
    
    def forward(self, x):
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 只对方阵进行变换
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, 'lora', lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            
            module.forward = forward_with_lora

def load_lora(model, path):
    # 从 path 加载保存好的 LoRA 权重字典，直接映射到 model 所在设备（避免设备不匹配）
    state_dict = torch.load(path, map_location=model.device)
    # 使用model = torch.nn.DataParallel(model)时，state_dict中的参数名会被自动加上'module.'前缀，因此需要去掉这个前缀才能正确加载多卡训练时得到的权重。
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # 遍历模型中所有被注入了 LoRA 适配器的 Linear 层
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 从 state_dict 中过滤出属于当前层 LoRA 的 key，并去掉层名前缀，
            # 得到仅包含 A/B 权重的局部字典（如 {'A.weight': ..., 'B.weight': ...}）
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 将提取出的权重加载到对应 LoRA 模块中
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    # 如果模型经过 torch.compile() 编译，会被包一层 _orig_mod，需要取出原始模型
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    # 遍历模型中所有被注入了 LoRA 适配器的 Linear 层
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 如果是 DataParallel 包装的模型，层名会带 'module.' 前缀，去掉以保持 key 的通用性
            clean_name = name[7:] if name.startswith("module.") else name
            # 将当前 LoRA 模块的权重（A.weight / B.weight）以 '{层名}.lora.{参数名}' 为 key 收集到总字典中
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    # 只保存 LoRA 参数（远小于全量权重），不保存冻结的原始模型参数
    torch.save(state_dict, path)
            