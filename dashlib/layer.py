import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
class LoRADashLayer:#只是一个工具，方便我们传参，然后定义self参数
    def __init__(
        self,
        r: int,
        alpha: Optional[float],
        dropout: float,
        merge_weights: bool,#表示是否愿意合并解合并
    ):
        self.r = r
        if alpha is None:
            self.scale = 1
        else:
            self.scale = alpha / r
        # Dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        # Merge state
        self.merged = False
        self.merge_weights = merge_weights

class LoRADashLinear(nn.Module, LoRADashLayer):#如果继承Linear层，我们需要重写reparameter方法，但是继承module不用，我们只需要__init__方法和forward方法，但是体现我们是linear层的特征的地方就是：foreard方法使用了 F.linear(input, self.weight, self.bias)，本质上和nn.Linear一样的他的forward也是用了这个方法
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        merge_weights: bool = True,
        bias: bool = False,
        
    ):
        # 初始化基类
        nn.Module.__init__(self)
        LoRADashLayer.__init__(
            self,
            r=r,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights
        )

        # 权重和偏置
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        # LoRA 和 Dash 参数
        if r > 0:
            # LoRA 参数
            self.loradash_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.loradash_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.weight.requires_grad = False

            # Dash 参数
            self.index = 8
            self.loradash_index = nn.Parameter(self.weight.new_zeros(self.index))
            self.weight_u_top = nn.Parameter(self.weight.new_zeros(out_features, self.index))
            self.weight_vt_top = nn.Parameter(self.weight.new_zeros(self.index, in_features))

            # 热身阶段参数
            self.warmup = 100
            self.FLAG = 0
            self.merge=False

    def loradash_init(
        self, weight_dtype: Optional[torch.dtype], adapter_dtype: Optional[torch.dtype]
    ):
        if weight_dtype is None:
            weight_dtype = self.weight.dtype
        if adapter_dtype is None:
            adapter_dtype = weight_dtype
        if hasattr(self, "loradash_A"):
            self.merged = False
            nn.init.kaiming_uniform_(self.loradash_A, a=math.sqrt(5))
            self.loradash_A.data = self.loradash_A.data.to(adapter_dtype)
            nn.init.zeros_(self.loradash_B).to(adapter_dtype)
            self.loradash_B.data = self.loradash_B.data.to(adapter_dtype)
            self.weight.data = self.weight.data.to(weight_dtype)


    def calculate_change_rate(self, a, bb, r):
        """计算变化率并返回 top-r 索引"""
        change_rate = torch.abs(bb) / torch.abs(a)
        _, top_r_indices = torch.topk(change_rate, r)
        return top_r_indices

    def _merge(self, mode: bool):#这段不对
        """合并或解合并权重"""
        if mode:
            if self.merge_weights and not self.merged and self.r > 0:
                merge = (self.loradash_B @ self.loradash_A) * self.scale
                self.weight.data += merge
                delta=(self.weight_u_top @ torch.diag(self.loradash_index) @ self.weight_vt_top)
                self.weight.data += delta
                self.merged = True
        else:
            if self.merge_weights and self.merged and self.r > 0:
                merge = (self.loradash_B @ self.loradash_A) * self.scale
                self.weight.data -= merge
                delta=(self.weight_u_top @ torch.diag(self.loradash_index) @ self.weight_vt_top)
                self.weight.data -= delta
                self.merged = False

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            # 基础线性变换
            result = F.linear(x, self.weight, bias=self.bias)
            # LoRA 增量
            loradash_delta = (self.loradash_B @ self.loradash_A) * self.scale
            result += self.dropout(x) @ loradash_delta.T

            # 热身阶段
            if self.FLAG < self.warmup:
                if self.FLAG == 0:
                    self.loradash_index.requires_grad = False
                    self.weight_u_top.requires_grad = False
                    self.weight_vt_top.requires_grad = False
                self.FLAG += 1
                return result
            
            # 热身结束，更新 Dash 参数
            elif self.FLAG == self.warmup:
                loradash_delta  = (self.loradash_B @ self.loradash_A) * self.scale# 计算LoRA增量矩阵
                weight_u, weight_sigma, weight_vt = torch.linalg.svd(self.weight.to(torch.float32), full_matrices=False)# 对原始权重矩阵进行SVD分解
                delta_sigma = torch.diag(torch.matmul(torch.matmul(weight_u.T, loradash_delta), weight_vt.T))
                top_index = self.calculate_change_rate(weight_sigma, delta_sigma, self.index)

                self.weight_u_top.data = weight_u[:, top_index]
                self.weight_vt_top.data = weight_vt[top_index, :]
                self.loradash_index.requires_grad = True
                self.FLAG += 1

            # 热身之后，加入 Dash 部分
            if self.FLAG > self.warmup:
                result += self.dropout(x) @ (self.weight_u_top @ torch.diag(self.loradash_index) @ self.weight_vt_top).T
                return result

        else:
            return F.linear(x, self.weight, bias=self.bias)
