#  ------------------------------------------------------------------------------------------
#  FOSSV: Singular Values and Orthonormal Regularized Singular Vectors Adaptation of Large Language Models
#  arXiv: https://arxiv.org/abs/2409.00055
#  Copyright (c) 2024 Yang Cao
#  Licensed under the Apache License, Version 2.0.
#  ------------------------------------------------------------------------------------------

"""
FOSSV model intergrated with Hugging Face transformers.
"""
import torch
import torch.nn as nn
from typing import List, Optional
from transformers import PreTrainedModel, AutoModelForCausalLM#这是含有model head（下游任务为因果的）的模型处理文件

from .layer import Linear as FOSSVLinear# 从自定义layer模块导入FOSSVLinear
from .config import FOSSVConfig # 从自定义config模块导入FOSSVConfig


class FOSSVModel(PreTrainedModel):#这个类最终形成了一个新的使用了FOSSV的模型
    """
    A wrapper model that applies FOSSV to huggingface PreTrainedModel.
    一个使用了fossv的PreTrainedModel形成的包装模型
    Attributes:
        config (FOSSVConfig): Configuration instance for this model.包括FOSSV的config参数
        model (PreTrainedModel): The wrapped PreTrainedModel.要包装的模型
    """

    config_class = FOSSVConfig

    def __init__(self, config):
        """
        Initialize the FOSSVModel.

        Args:
            config (FOSSVConfig): Configuration for the FOSSV model.
        """
        super().__init__(config)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, trust_remote_code=True
        )#第二个参数表示是否信任从远程加载的模型代码。
        self._replace_modules()

    def _set_submodule(self, target: str, module: torch.nn.Module):#这个自定义函数是用于查找替换一个模块
        """
        pytorch中每个层都要继承nn.Module类，所以要设置子模块。
        Set the submodule given by ``target`` if it exists, otherwise throw an error.
        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:
        .. code-block:: text
            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )
        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)
        To overide the ``Conv2d`` with a new submodule ``Linear``, you
        would call                                                       为了使用一个线形层子模块来代替``Conv2d``，你可以调用set_submodule函数
        ``set_submodule("net_b.net_c.conv", nn.Linear(33, 16))``.
        Args:   set_submodule函数的参数解释
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)     目标模块名称，是一个完全量化的字符串，比如``net_b.net_c.conv``。
            module: The module to set the submodule to.
        Raises:                              会出现报错的情况
            ValueError: If the target string is empty
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        atoms: List[str] = target.split(".") # 将目标字符串按"."分割，得到模块名称，形成一个list列表
        name = atoms.pop(-1) # 获取模块的最后一个部分（即模块的名称）
        mod = self # 初始化自身完整的模型为mod，下面会逐渐缩减，相当于是根节点到子节点

        for item in atoms:# 遍历目标路径的每一部分
            if not hasattr(mod, item):# 如果当前模块没有此属性，抛出错误
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item) # 如果self自身这个模型有的话，则获取当前模块的子模块

            if not isinstance(mod, torch.nn.Module):# 如果找到的不是一个nn.Module类型，抛出错误
                raise AttributeError("`" + item + "` is not an nn.Module")
        #两个if not保证了获取的mod是nn.Module类型
        setattr(mod, name, module)# setattr(object, name, value)object是要修改的对象，name是要修改的object的具体的属性或者说子模块，value是要修改的目标


    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.

        Returns:
            The output of the wrapped model's forward pass.
        """
        return self.model(*args, **kwargs)# 执行预训练模型的前向传播

    def _replace_modules(self):
        """
        Replace linear layers in target_modules with FOSSV enabled Linear.将目标模块的线性层替换为FOSSV启用的线性层。
        """
        if isinstance(self.config.target_modules, list):#检查我们输入的想要改变的模块是否是一个list
            target_module_names = self.config.target_modules
        elif isinstance(self.config.target_modules, dict):#如果是一个字典，我们将其转化为list，转化为list后方便我们后续调用_set_submodule
            target_module_names = list(self.config.target_modules.values())
        else:
            raise ValueError("target_modules must be a list or dict")
        for name, module in self.named_modules(): # 遍历模型的所有模块同时返回模块的名称和模块本身
            if any(t in name for t in target_module_names) and isinstance(
                module, nn.Linear
            ):# 如果模块名称符合target是我们想要改变的且是线性层
                fossv_module = FOSSVLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=self.config.rank,
                    alpha=self.config.alpha,
                    bias=module.bias is not None,
                    dropout=self.config.dropout,
                    mod=self.config.mod,
                )#输入参数创建模块，创建一个FOSSVLinear对象即使使用fossv的线性层
                fossv_module.weight.data = module.weight.data # 将权重数据复制
                if fossv_module.bias is not None:
                    fossv_module.bias.data = module.bias.data# 将偏置数据复制
                self._set_submodule(f"{name}", fossv_module)# 替换模块，使用新的模块替换旧的模块

    def fossv_init(
        self,
        weight_dtype: Optional[torch.dtype] = None,
        adapter_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize FOSSV adapters for all FOSSV enabled Linear layers in the model.

        Args:
            weight_dtype (Optional[torch.dtype]): Data type for the weight matrix.
            adapter_dtype (Optional[torch.dtype]): Data type for the FOSSV matrices.
        """
        print("Initializing FOSSV Adapters...")# 输出初始化信息
        for module in self.modules():# 遍历所有模块
            if isinstance(module, FOSSVLinear): # 如果模块是FOSSVLinear类型
                module.fossv_init(weight_dtype, adapter_dtype)# 初始化FOSSV适配器

    def merge(self, mode=True):
        """
        Merge or unmerge all FOSSV adapters in the model.

        Args:
            mode (bool): If True, merge the weights. If False, unmerge the weights.  mode是一个bool变量，表示是否要合并权重。
        """
        for module in self.modules():
            if isinstance(module, FOSSVLinear):
                module._merge(mode)

    def get_parameters(self) -> List[nn.Parameter]:
        """
        Get all FOSSV adapters in the model.

        Returns:
            List[nn.Parameter]: List of all parameters with 'fossv_' in their name.
        """
        return [p for n, p in self.named_parameters() if "fossv_" in n] # 获取所有带有"fossv_"的参数

    def set_trainable(self, mode=True):
        """
        Set the trainable state of all FOSSV adapters.

        Args:
            mode (bool): If True, make FOSSV adapters trainable. If False, freeze them.
        """
        for name, param in self.named_parameters():# 遍历所有参数
            if "fossv_" in name: # 如果参数名称中包含"fossv_"
                param.requires_grad = mode# 设置可训练状态
            else:
                param.requires_grad = False# 设置不可训练状态
