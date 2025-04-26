import torch

def inspect_adapters(checkpoint_path):
    # 加载模型参数
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # 定义适配器标识模式
    LORA_KEYS = {'lora_A', 'lora_B', 'scaling'}
    DASH_KEYS = {'lora_index', 'weight_u_top', 'weight_vt_top'}
    
    # 自动识别适配器类型
    adapters = {}
    for key in state_dict.keys():
        # 识别LoRA适配器
        if any(lora_key in key for lora_key in LORA_KEYS):
            layer_name = key.rsplit('.', 1)[0]  # 提取层名称
            adapters.setdefault(layer_name, {'type': []})['type'].append('LoRA')
            
        # 识别DASH适配器
        if any(dash_key in key for dash_key in DASH_KEYS):
            layer_name = key.rsplit('.', 1)[0]
            adapters.setdefault(layer_name, {'type': []})['type'].append('DASH')
            
    # 整理显示结果
    print("🔍 发现以下适配器结构：")
    for layer, info in adapters.items():
        types = list(set(info['type']))  # 去重
        num_params = sum(1 for k in state_dict if k.startswith(layer))
        
        # 显示适配器详细信息
        print(f"├── 层: {layer}")
        print(f"│   ├── 类型: {', '.join(types)}")
        print(f"│   ├── 参数量: {num_params}个")
        print(f"│   └── 具体参数: ")
        for param in [k for k in state_dict if k.startswith(layer)]:
            shape = tuple(state_dict[param].shape)
            print(f"│       ├── {param}: {shape}")

if __name__ == "__main__":
    # 使用示例
    inspect_adapters("your_model_checkpoint.pth")