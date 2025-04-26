import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional


class LoRALayer:
    def __init__(
        self,
        r: int,
        alpha: Optional[float],
        dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.alpha = alpha
        if alpha == None:
            self.scale = 1
        else:
            self.scale = alpha / r
        # Optional dropout
        if dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Module, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        merge_weights: bool = True,
        bias=False,
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(
            self,
            r=r,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights,
        )

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias == True:
            self.bias = nn.Parameter(torch.empty((out_features)))
        else:
            self.bias = None
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

    def pissa_init(
        self, weight_dtype: Optional[torch.dtype], adapter_dtype: Optional[torch.dtype]
    ):
        if weight_dtype is None:
            weight_dtype = self.weight.dtype
        if adapter_dtype is None:
            adapter_dtype = weight_dtype
        if hasattr(self, "lora_A"):
            self.merged = False
            self.weight.data.to(torch.float32)  # Convert to float32 for SVD
            u, s, vt = torch.linalg.svd(self.weight.T, full_matrices=False)
            s_r = s[: self.r]
            self.lora_A.data = (
                (u[:, : self.r] @ torch.diag(s_r**0.5)).T.contiguous().to(adapter_dtype)
            )
            self.lora_B.data = (
                (torch.diag(s_r**0.5) @ vt[: self.r, :])
                .T.contiguous()
                .to(adapter_dtype)
            )
            merge = self.lora_B @ self.lora_A
            self.weight.data = (self.weight - merge * self.scale).to(weight_dtype)

    def milora_init(#新增修改
        self, weight_dtype: Optional[torch.dtype], adapter_dtype: Optional[torch.dtype]#指定适配器的参数类型
    ):
        if weight_dtype is None:
            weight_dtype = self.weight.dtype
        if adapter_dtype is None:
            adapter_dtype = weight_dtype
        if hasattr(self, "lora_A"):#检查一下模型结构中是否存在属性lora_A
            self.merged = False#merge属性标志是否完成了适配器参数的合并，因为下面我们要将适配器从权重矩阵中分离出来，所以最后的状态就是没有合并，设置为False
            #self.weight.to(torch.float32)  # Convert to float32 for SVD，因为只有float32参数类型才能svd分解
            weight_fp32 = self.weight.to(torch.float32)  # 创建临时float32副本
            u, s, vt = torch.linalg.svd(weight_fp32.T, full_matrices=False)
            s_r = s[-self.r:]  # Store s_small on the correct device

            self.lora_A.data = (
                (u[:, -self.r:]  @ torch.diag(s_r**0.5)).T.contiguous().to(adapter_dtype)
            )
            self.lora_B.data = (
                (torch.diag(s_r**0.5) @ vt[-self.r:, :])
                .T.contiguous()
                .to(adapter_dtype)
            )
            merge = self.lora_B @ self.lora_A
            self.weight.data = (self.weight - merge * self.scale).to(weight_dtype)



    def lora_init(
        self, weight_dtype: Optional[torch.dtype], adapter_dtype: Optional[torch.dtype]
    ):
        if weight_dtype is None:
            weight_dtype = self.weight.dtype
        if adapter_dtype is None:
            adapter_dtype = weight_dtype
        if hasattr(self, "lora_A"):
            self.merged = False
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.data = self.lora_A.data.to(adapter_dtype)
            nn.init.zeros_(self.lora_B).to(adapter_dtype)
            self.lora_B.data = self.lora_B.data.to(adapter_dtype)
            self.weight.data = self.weight.data.to(weight_dtype)

    def merge(self, mode: bool):
        if mode:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += self.lora_B @ self.lora_A * self.scale
                self.merged = True
        else:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= self.lora_B @ self.lora_A * self.scale
                self.merged = False

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            result += (
                F.linear(self.lora_dropout(x), self.lora_B @ self.lora_A) * self.scale
            )
            if self.bias is not None:
                result += self.bias
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)
