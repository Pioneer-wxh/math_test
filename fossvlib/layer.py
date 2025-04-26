import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class FOSSVLayer:
    def __init__(
        self,
        r: int,
        alpha: Optional[float],
        dropout: float,
        merge_weights: bool,  # 用来控制是否进行合并解合并
        mod: str = "small",  # 新增 mod 参数
    ):
        super().__init__()
        self.mod = mod  # 记录模式
        self.r = r
        if alpha is None:
            self.scale = 1
        else:
            self.scale = alpha / r
        # Optional dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False  # 这个布尔变量是用来表示内部状态，模型是否合并
        self.merge_weights = merge_weights

class Linear(nn.Module, FOSSVLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        merge_weights: bool = True,
        bias=False,
        mod: str = "small",
    ):
        # 先初始化nn.Linear
        nn.Module.__init__(self)

        FOSSVLayer.__init__(
            self,
            r=r,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights,
            mod=mod

        )

        # 创建必要的参数容器，便于后面的初始化储存原始linear层的参数
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))  # 初始化权重矩阵，初始化一个随机的权重矩阵并且转化为pytorch版本
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features)))  # 如果有bias的话，初始化一个bias，是一个一维张量
        else:
            self.bias = None

        

        if r > 0:
            self.fossv_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.fossv_S = nn.Parameter(self.weight.new_zeros(r))
            self.fossv_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.weight.requires_grad = False
           
            
            # 注册为空的张量（避免None）
            self.register_buffer("U", torch.empty(in_features, min(in_features, out_features)))
            self.register_buffer("S", torch.empty(min(in_features, out_features)))
            self.register_buffer("Vt", torch.empty((min(in_features, out_features), out_features)))

            # Dash part adapter preparation
            self.index = 8
            self.fossv_index  = nn.Parameter(self.weight.new_zeros(self.index))
            self.weight_u_top = nn.Parameter(self.weight.new_zeros(in_features, self.index))
            self.weight_vt_top = nn.Parameter(self.weight.new_zeros(self.index, out_features))

            self.warmup = 100
            self.FLAG = 0

    def fossv_init(self, weight_dtype=None, adapter_dtype=None):
        if not hasattr(self, "fossv_A"):
            return

        weight_dtype = weight_dtype or self.weight.dtype
        adapter_dtype = adapter_dtype or weight_dtype

        # SVD分解并强制连续化
        u, s, vt = torch.linalg.svd(self.weight.T.float(), full_matrices=False)
        self.U.data = u.detach().clone().contiguous()  # 确保连续
        self.S.data = s.detach().clone().contiguous()
        self.Vt.data = vt.detach().clone().contiguous()

        # 截取子矩阵并强制连续化
        if self.mod == "large":
            u_sub = self.U[:, :self.r].detach().clone().contiguous()
            s_sub = self.S[:self.r].detach().clone().contiguous()
            vt_sub = self.Vt[:self.r, :].detach().clone().contiguous()
        elif self.mod == "random":
            indices = torch.randperm(len(self.S))[:self.r]
            u_sub = self.U[:, indices].detach().clone().contiguous()
            s_sub = self.S[indices].detach().clone().contiguous()
            vt_sub = self.Vt[indices, :].detach().clone().contiguous()
        else:
            u_sub = self.U[:, -self.r:].detach().clone().contiguous()
            s_sub = self.S[-self.r:].detach().clone().contiguous()
            vt_sub = self.Vt[-self.r:, :].detach().clone().contiguous()

        # 适配器参数赋值（确保连续）
        self.fossv_A.data = u_sub.T.contiguous().to(adapter_dtype)
        self.fossv_S.data = s_sub.contiguous().to(adapter_dtype)
        self.fossv_B.data = vt_sub.T.contiguous().to(adapter_dtype)

        # Dash部分参数连续化
        self.weight_u_top.data = self.weight_u_top.data.contiguous()
        self.weight_vt_top.data = self.weight_vt_top.data.contiguous()

        # 合并适配器
        merge = (self.fossv_B @ torch.diag(self.fossv_S)) @ self.fossv_A
        self.weight.data = (self.weight - merge * self.scale).to(weight_dtype)


    def calculate_change_rate(self, a, bb, r):

        change_rate = abs(bb) / abs(a)
        _, top_r_indices = torch.topk(change_rate, r)
        return top_r_indices

    def _merge(self, mode: bool):  # mode表示是否要进行合并的动作，self.merged表示状态
        """
        Merge or unmerge FOSSV weights with the main weight matrix.是否和主权重矩阵进行合并

        Args:
            mode (bool): If True, merge the weights. If False, unmerge the weights.根据mode变量决定如何进行操作
        """
        if mode:  # 合并
            if self.merge_weights and not self.merged:  # 如果要进行合并，且模型未进行合并
                # Merge the weights and mark it
                if self.r > 0:
                    merge = (self.fossv_B @ torch.diag(self.fossv_S)) @ self.fossv_A
                    self.weight.data += merge * self.scale
                    merge_fossv = (self.weight_u_top @ torch.diag(self.fossv_index) @ self.weight_vt_top)
                    self.weight.data += merge_fossv.T
                
                self._buffers.pop("U", None)
                self._buffers.pop("S", None)
                self._buffers.pop("Vt", None)
                
                self.merged = True
        else:  # 解合并
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    merge = (self.fossv_B @ torch.diag(self.fossv_S)) @ self.fossv_A
                    self.weight.data -= merge * self.scale
                    merge_fossv = (self.weight_u_top @ torch.diag(self.fossv_index) @ self.weight_vt_top)
                    self.weight.data -= merge_fossv.T
                self.merged = False

    def forward(self, x: torch.Tensor):
        


        if self.r > 0 and not self.merged:  # r大于0，且当前模型未进行合并
            result = F.linear(x, self.weight, bias=self.bias)
            fossv_delta = (self.fossv_B @ torch.diag(self.fossv_S)) @ self.fossv_A * self.scale
            result += self.dropout(x) @ fossv_delta.T

            if self.FLAG < self.warmup:
                if self.FLAG == 0:
                    self.fossv_index .requires_grad = False
                    self.weight_u_top.requires_grad = False
                    self.weight_vt_top.requires_grad = False
                self.FLAG += 1
                return result

            # layer.py 的 forward 方法
            elif self.FLAG == self.warmup:
                # 确保 u 和 vt 在正确的设备上
                U = self.U
                Vt = self.Vt
                
                # 计算 delta_sigma（确保 fossv_delta 在正确设备上）
                fossv_delta = fossv_delta
                # 修正步骤：添加转置操作
                delta_sigma = torch.diag(self.U.T @ fossv_delta.T @ self.Vt.T)
                            
                
                # 确保 self.s 在正确设备上
                top_index = self.calculate_change_rate(self.S, delta_sigma, self.index)

                # Update SVD decomposition U and Vt matrices
                self.weight_u_top.data = U[:, top_index]
                self.weight_vt_top.data = Vt[top_index, :]

                # Unfreeze dash part parameters
                self.fossv_index.requires_grad = True
                self.FLAG += 1

            if self.FLAG > self.warmup:
                result += self.dropout(x) @ (self.weight_u_top @ torch.diag(self.fossv_index) @ self.weight_vt_top)
                return result

        else:
            return F.linear(x, self.weight, bias=self.bias)




