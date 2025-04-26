import torch
import torch.nn as nn
from typing import List, Optional
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
)
from .config import LoRADashConfig  # 修正1：修正配置类名称
from .layer import LoRADashLinear  # 修正2：修正线性层导入


class LoRADashModel(PreTrainedModel):#我们自定义了lora模型的一些方法，方便后续用LoRAModel拆功能键模型后调用函数进行操作。这就是继承PreTrainedModel的好处，有些自定义的方法方便我们对模型进行处理
    config_class = LoRADashConfig

    def __init__(self, config):#魔法方法，会直接执行
        super().__init__(config)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, trust_remote_code=True
        )
        self._replace_modules()

    def _set_submodule(self, target: str, module: torch.nn.Module):#替换模块的具体工具
        atoms: List[str] = target.split(".")# 将 target 按点号分割成字符串列表，例如 "layer1.conv2" 变成 ["layer1", "conv2"]
        name = atoms.pop(-1)# 移除并返回 atoms 的最后一个元素作为目标子模块名，例如 atoms=["layer1"]，name="conv2"
        mod = self# 将当前模型实例 self 赋值给 mod，作为遍历的起点，就像是树的主节点

        for item in atoms:#遍历名称列表，目的是一层一层的逼近目标模块
            if not hasattr(mod, item):# 检查 mod 是否有名为 item 的属性，如果没有就抛出异常
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)#如果存在# 获取 mod 的 item 属性并更新 mod，例如 mod=self.layer1，相当于是递进一层，接近目标

            if not isinstance(mod, torch.nn.Module):# 检查 mod 是否是 nn.Module 类型，如果不是抛出异常
                raise AttributeError("`" + item + "` is not an nn.Module")

        setattr(mod, name, module)#按此方法遍历，最后到达最后一层后将 module 赋值给 mod 的 name 属性，完成替换

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)#向前传播出个传递参数

    def _replace_modules(self):#对模型执行替换，将原始的线性层替换为LoRALinear
        #总之要将目标变为列表类型
        if isinstance(self.config.target_modules, list):# 检查 self.config.target_modules 是否是列表类型
            target_module_names = self.config.target_modules# 如果是列表，直接赋值给 target_module_names，例如 ["linear1", "linear2"]
        elif isinstance(self.config.target_modules, dict):# 检查 self.config.target_modules 是否是字典类型
            target_module_names = list(self.config.target_modules.values())# 如果是字典，将字典的值转为列表赋值给 target_module_names，例如 {"key1": "linear1"} -> ["linear1"]
        else:
            raise ValueError("target_modules must be a list or dict")
        for name, module in self.named_modules():# 遍历模型的所有子模块，name 是模块的名称（字符串），module 是模块实例。模型其实就是一个一个模块嵌套构建
            if any(t in name for t in target_module_names) and isinstance(
                module, nn.Linear#如果找到了目标模块并且类型符合标准的话，进行替换
            ):
                loradash_module = LoRADashLinear(  # 修正4：使用正确的类名
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=self.config.rank,
                    alpha=self.config.alpha,
                    bias=module.bias is not None,
                    dropout=self.config.dropout,
                )
                loradash_module.weight.data = module.weight.data
                if loradash_module.bias is not None:
                    loradash_module.bias.data = module.bias.data
                self._set_submodule(f"{name}", loradash_module)# 调用 _set_submodule 方法，将模型中名为 name 的模块替换为新的 lora_module
                #f"{name}" 的作用是生成 _set_submodule 方法所需的第一个参数 target，即目标模块的路径，f"{name}" 的结果就是 name 的字符串值本身，例如 "layer1.linear2"。
#先替换层，有层之后才能调用我们自定义的方法
    def loradash_init(#这是继承PreTrainedModel后自定义的方法，相当于预训练模型本身就有这个方法，方便我们直接对模型进行操作
        self,
        weight_dtype: Optional[torch.dtype] = None,
        adapter_dtype: Optional[torch.dtype] = None,
    ):
        print("Initializing LoRADash Adapters...")
        for module in self.modules():
            if isinstance(module, LoRADashLinear):#新增修改
                module.loradash_init(weight_dtype, adapter_dtype) 
                    
                    
                    
    def merge(self, mode=True):
        for module in self.modules():
            if isinstance(module, LoRADashLinear):
                module._merge(mode)

    def get_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "loradash_" in n]

    def set_trainable(self, mode=True):
        for name, param in self.named_parameters():
            if "loradash_" in name:
                param.requires_grad = mode
            else:
                param.requires_grad = False
