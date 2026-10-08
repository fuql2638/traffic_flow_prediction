# ============================================================
# MBSTGFormer - Stage 1 Code-Level Optimized Version
# 基于用户原始 MBSTGFormer.py 修改
# 重点：edge-only 动态图、Sparse GAT/GCN、图缓存、删除无效参数、若干正确性修复
# ============================================================
from typing import Tuple, Optional
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import math
import warnings
import torch.nn.functional as F
from typing import Tuple
import os
import time
from datetime import datetime, timedelta
import json
# 历史图使用归一化的训练数据做，解码器使用归一化的数据
# 设置中文字体支持
import matplotlib as mpl
from scipy.interpolate import interp1d

# 设置随机种子确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

class SpatioTemporalEmbed(nn.Module):
    """基于真实地理邻接图的 Laplacian 空间特征 + 周期时间特征 → 时空嵌入 M。

    第一阶段优化：
    1. 原代码在 __init__ 中用单位矩阵构造 Laplacian，空间嵌入与真实路网无关；
       这里改为首次 forward 时基于 adj_geo 计算，并缓存结果。
    2. 空间谱分解只执行一次，不参与反向传播。
    """

    def __init__(self, num_nodes, d_model, max_len=5000, spatial_k=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.spatial_k = min(spatial_k, max(1, num_nodes - 1))

        # 非持久缓存：保存由真实地理图计算得到的 Laplacian 特征向量
        self.register_buffer('spatial', torch.empty(0), persistent=False)

        # 时间：小时 + 星期 one-hot → MLP
        self.temp_mlp = nn.Sequential(
            nn.Linear(24 + 7, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 空间 + 时间融合
        self.fusion = nn.Sequential(
            nn.Linear(self.spatial_k + d_model, d_model),
            nn.Sigmoid()
        )

    @torch.no_grad()
    def _build_spatial_embedding(self, adj_geo: torch.Tensor):
        if adj_geo is None:
            raise ValueError('SpatioTemporalEmbed 需要 adj_geo 以构造真实路网空间嵌入。')

        A = adj_geo.detach().float()
        if A.dim() == 3:
            A = A[0]
        if A.dim() != 2 or A.size(0) != A.size(1):
            raise ValueError(f'adj_geo 应为 [N,N]，实际形状为 {tuple(A.shape)}')

        # 保证数值对称
        A = (A + A.t()) * 0.5
        N = A.size(0)
        deg = A.sum(dim=1).clamp_min(1e-8)
        deg_inv_sqrt = deg.rsqrt()
        L = torch.eye(N, device=A.device, dtype=A.dtype) - \
            deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]

        _, eig_vec = torch.linalg.eigh(L)

        if N > 1:
            spatial = eig_vec[:, 1:1 + self.spatial_k]
        else:
            spatial = torch.zeros((N, 0), device=A.device, dtype=A.dtype)

        # 极小图兼容：不足 spatial_k 时补 0
        if spatial.size(1) < self.spatial_k:
            pad = torch.zeros(
                N, self.spatial_k - spatial.size(1),
                device=A.device, dtype=A.dtype
            )
            spatial = torch.cat([spatial, pad], dim=1)

        self.spatial = spatial.contiguous()

    def forward(self, x_mark_enc, adj_geo=None):
        """
        x_mark_enc: [B,T,4]，小时/星期/月/周末（归一化）
        adj_geo: [N,N]
        返回 M: [B,T,N,D]，值域 0~1
        """
        B, T, _ = x_mark_enc.shape

        # 首次调用或设备变化时构造一次空间嵌入
        if self.spatial.numel() == 0 or self.spatial.device != x_mark_enc.device:
            self._build_spatial_embedding(adj_geo.to(x_mark_enc.device))

        h = (x_mark_enc[..., 0] * 23).round().clamp(0, 23).long()
        w = (x_mark_enc[..., 1] * 6).round().clamp(0, 6).long()
        h_one = F.one_hot(h, 24).float()
        w_one = F.one_hot(w, 7).float()
        t_emb = self.temp_mlp(torch.cat([h_one, w_one], dim=-1))  # [B,T,D]

        s_emb = self.spatial.unsqueeze(0).expand(B, -1, -1)       # [B,N,K]
        t_broad = t_emb.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        s_broad = s_emb.unsqueeze(1).expand(-1, T, -1, -1)
        M = self.fusion(torch.cat([t_broad, s_broad], dim=-1))
        return M


class TrendSeasonEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gru_t = nn.GRU(d_model, d_model, batch_first=True)
        self.gru_s = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, xt, xs):
        """
        xt/xs: [B*T,N,D]  已reshape
        返回: [B*T,N,D]  趋势+季节
        """
        yt, _ = self.gru_t(xt)  # 趋势通道
        ys, _ = self.gru_s(xs)  # 季节通道
        return yt + ys  # 元素级合并，与STDN一致
        
# === 新增：趋势季节分解模块 ===
class TrendSeasonalDecomposition(nn.Module):
    """时空感知趋势-季节分解。

    原代码还计算了一个 result = vector*trend + (1-vector)*seasonal，
    但主模型从未使用 result，因此 vector 是无效参数、result 是无效中间张量。
    第一阶段直接删除，保持当前有效预测路径不变。
    """

    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes

    def forward(self, X, STEmbedding):
        trend = X * STEmbedding
        seasonal = X - trend
        return trend, seasonal

class ProbSparseSelfAttention(nn.Module):
    """ProbSparse自注意力机制 - 原始论文实现"""

    def __init__(self, d_model: int, n_heads: int = 8, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.dropout = nn.Dropout(dropout)

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor):
        """ProbSparse查询-键采样"""
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 采样部分键
        sample_k = min(max(1, L_K // self.factor), L_K)
        indices = torch.randint(0, L_K, (sample_k,), device=Q.device)
        K_sample = K[:, :, indices, :]

        # 计算查询-采样键的相似度
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))

        # 计算重要性度量
        M = Q_K_sample.max(dim=-1, keepdim=True)[0] - torch.div(Q_K_sample.sum(dim=-1, keepdim=True), L_K)
        return M.squeeze(-1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        B, L_Q, d_model = queries.shape
        L_K = keys.size(1)

        # 线性投影 + 多头分割
        Q = self.w_q(queries).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(keys).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(values).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        # ProbSparse采样
        M = self._prob_QK(Q, K)
        u = min(self.factor * int(math.ceil(math.log(L_K))), L_Q)

        # 选择重要查询
        indices = M.topk(u, dim=-1)[1]

        # 采样重要查询
        batch_indices = torch.arange(B, device=queries.device)[:, None, None]
        head_indices = torch.arange(self.n_heads, device=queries.device)[None, :, None]

        Q_reduce = Q[batch_indices, head_indices, indices]

        # 计算稀疏注意力
        attn_scores = torch.matmul(Q_reduce, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文
        context = torch.matmul(attn_weights, V)

        # 恢复完整上下文
        context_full = torch.zeros(B, self.n_heads, L_Q, self.d_k, device=queries.device)
        context_full[batch_indices, head_indices, indices] = context

        # 恢复原始维度
        context_full = context_full.transpose(1, 2).contiguous().view(B, L_Q, d_model)
        output = self.w_o(context_full)

        return output, attn_weights


class InformerEncoderLayer(nn.Module):
    """Informer编码器层"""

    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        # ProbSparse自注意力
        self.self_attention = ProbSparseSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 前馈网络（使用1D卷积实现）
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # 自注意力子层
        residual = x
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(residual + self.dropout1(attn_out))

        # 前馈网络子层
        residual = x
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        x = self.norm2(residual + self.dropout2(x))

        return x


class ConvDistill(nn.Module):
    """卷积蒸馏层"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='replicate')
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [batch, seq_len, d_model]
        residual = x

        # 卷积 + 池化下采样
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        x = x.transpose(1, 2)

        # 残差连接（下采样匹配）
        residual = residual[:, ::2, :]  # 步长为2的下采样

        return self.norm(x + residual)


class InformerEncoder(nn.Module):
    """完整的Informer编码器"""

    def __init__(self, d_model: int, n_layers: int = 3, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1, use_distill: bool = True):
        super().__init__()

        self.use_distill = use_distill
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 蒸馏层
        if use_distill and n_layers > 1:
            self.distill_layers = nn.ModuleList([
                ConvDistill(d_model, dropout) for _ in range(n_layers - 1)
            ])
        else:
            self.distill_layers = None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # 编码器处理
        if self.distill_layers is not None:
            for i, layer in enumerate(self.encoder_layers):
                x = layer(x)
                if i < len(self.encoder_layers) - 1:  # 最后一层不蒸馏
                    x = self.distill_layers[i](x)
        else:
            for layer in self.encoder_layers:
                x = layer(x)

        return self.norm(x)


class InformerEncoderWithBiLSTM(nn.Module):
    """集成BiLSTM的Informer编码器"""

    def __init__(self, d_model: int, n_layers: int = 3, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1, use_distill: bool = True,
                 lstm_hidden: int = 256, lstm_layers: int = 2):
        super().__init__()

        self.use_distill = use_distill
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 蒸馏层
        if use_distill and n_layers > 1:
            self.distill_layers = nn.ModuleList([
                ConvDistill(d_model, dropout) for _ in range(n_layers - 1)
            ])
        else:
            self.distill_layers = None

        # BiLSTM模块 - 在编码器中融合
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # 维度适配层
        self.dim_adapter = nn.Linear(lstm_hidden * 2, d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # 编码器处理
        if self.distill_layers is not None:
            for i, layer in enumerate(self.encoder_layers):
                x = layer(x)
                if i < len(self.encoder_layers) - 1:  # 最后一层不蒸馏
                    x = self.distill_layers[i](x)
        else:
            for layer in self.encoder_layers:
                x = layer(x)

        # BiLSTM处理 - 增强序列建模
        lstm_out, _ = self.bilstm(x)  # [batch_size, seq_len, lstm_hidden*2]
        lstm_out = F.relu(self.dim_adapter(lstm_out))  # [batch_size, seq_len, d_model]

        # 残差连接：结合Informer和BiLSTM的输出
        x = x + lstm_out

        return self.norm(x)


class InformerDecoderLayer(nn.Module):
    """Informer解码器层"""

    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        # 掩码自注意力
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor = None):
        # 掩码自注意力
        residual = x
        self_attn_out, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(residual + self.dropout1(self_attn_out))

        # 交叉注意力
        residual = x
        cross_attn_out, _ = self.cross_attention(x, enc_output, enc_output)
        x = self.norm2(residual + self.dropout2(cross_attn_out))

        # 前馈网络
        residual = x
        ffn_out = self.ffn(x)
        x = self.norm3(residual + self.dropout3(ffn_out))

        return x


class LSTMDecoder(nn.Module):
    """BiLSTM增强的解码器"""

    def __init__(self, d_model: int, pred_len: int, n_layers: int = 2,
                 n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1,
                 lstm_hidden: int = 256, lstm_layers: int = 1):
        super().__init__()

        self.pred_len = pred_len
        self.decoder_layers = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # BiLSTM增强模块
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.lstm_adapter = nn.Linear(lstm_hidden, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成因果掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor):
        # 生成掩码
        tgt_mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)

        # 解码器处理
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask)

        # BiLSTM增强
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, lstm_hidden*2]
        lstm_out = F.relu(self.lstm_adapter(lstm_out))  # [batch_size, seq_len, d_model]

        # 残差连接
        x = x + lstm_out

        x = self.norm(x)

        # 修改：对每个时间步进行投影
        # 只取后pred_len个时间步（解码器的预测部分）
        pred_slice = x[:, -self.pred_len:, :]  # [batch_size, pred_len, d_model]
        # output = self.projection(pred_slice)  # [batch_size, pred_len, 1]
        # output = output.squeeze(-1)  # [batch_size, pred_len]

        # 对每个时间步进行投影，得到单个值
        batch_size, pred_len, d_model = pred_slice.shape
        pred_slice_flat = pred_slice.reshape(batch_size * pred_len, d_model)
        output_flat = self.output_projection(pred_slice_flat)  # [batch_size*pred_len, 1]
        output = output_flat.reshape(batch_size, pred_len)  # [batch_size, pred_len]

        return output


# === 1) 修正的 PositionalEncoding ===
class PositionalEncoding(nn.Module):
    """标准位置编码，输出与输入形状一致 [batch, seq_len, d_model]"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.dtype)


class PositionalEmbedding(nn.Module):
    """位置编码 - 使用可学习的位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TokenEmbedding(nn.Module):
    """令牌嵌入 - 用于数值特征的嵌入"""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, c_in]
        x = x.transpose(1, 2)  # [batch_size, c_in, seq_len]
        x = self.token_conv(x)
        x = x.transpose(1, 2)  # [batch_size, seq_len, d_model]
        return x


class DataEmbedding(nn.Module):
    """数据嵌入 - 结合数值特征和位置编码"""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


def dense_adj_to_edges(adj: torch.Tensor, eps: float = 0.0):
    """将固定邻接矩阵转换为稀疏边表示。

    约定 edge_index[0] 为目标节点 i（邻接矩阵行），edge_index[1] 为源节点 j（列），
    对应 dense attention 中 scores[..., i, j]。
    """
    if not torch.is_tensor(adj):
        adj = torch.as_tensor(adj, dtype=torch.float32)
    if adj.dim() == 3:
        # 静态图应对整个 batch 相同；兼容 [1,N,N]
        adj = adj[0]
    if adj.dim() != 2:
        raise ValueError(f'adj 必须为 [N,N]，实际为 {tuple(adj.shape)}')

    mask = adj.abs() > eps
    dst, src = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([dst, src], dim=0).long()
    edge_weight = adj[dst, src].float()
    return edge_index, edge_weight


def _edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int, eps: float = 1e-12):
    """按目标节点分组的 edge softmax。

    scores: [B,H,E]
    dst: [E]
    return: [B,H,E]
    """
    B, H, E = scores.shape
    index = dst.view(1, 1, E).expand(B, H, E)

    max_per_node = torch.full(
        (B, H, num_nodes), -torch.inf,
        device=scores.device, dtype=scores.dtype
    )
    max_per_node.scatter_reduce_(2, index, scores, reduce='amax', include_self=True)
    stabilized = scores - max_per_node.gather(2, index)
    exp_scores = stabilized.exp()

    denom = torch.zeros(
        (B, H, num_nodes), device=scores.device, dtype=scores.dtype
    )
    denom.scatter_add_(2, index, exp_scores)
    return exp_scores / denom.gather(2, index).clamp_min(eps)


class SparseMultiHeadGraphAttention(nn.Module):
    """Edge-only 多头图注意力。

    与原 Dense GAT 的主要区别：不再构造 [B,H,N,N] attention matrix，
    仅对图中实际存在的 E 条边计算注意力，复杂度由 O(N^2) 降为 O(E)。
    """

    def __init__(self, in_features, out_features, n_heads, dropout=0.1):
        super().__init__()
        if out_features % n_heads != 0:
            raise ValueError('out_features 必须能被 n_heads 整除')
        self.n_heads = n_heads
        self.out_features = out_features
        self.d_k = out_features // n_heads

        self.w_q = nn.Linear(in_features, out_features)
        self.w_k = nn.Linear(in_features, out_features)
        self.w_v = nn.Linear(in_features, out_features)
        self.w_o = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None, use_edge_weight=False):
        """
        x: [B,N,F]
        edge_index: [2,E]，0=dst, 1=src
        edge_weight: [E] 或 [B,E]
        use_edge_weight: 动态图为 True，将动态图权重作为 attention prior。
        """
        B, N, _ = x.shape
        dst, src = edge_index[0], edge_index[1]
        E = dst.numel()

        Q = self.w_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        q_dst = Q[:, :, dst, :]  # [B,H,E,D]
        k_src = K[:, :, src, :]
        v_src = V[:, :, src, :]

        scores = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.d_k)  # [B,H,E]

        # 原 Dense GAT 只使用 adj 做 mask，实际没有利用动态边权强弱。
        # 对动态图开启该项，使论文中的“动态关联强度”真正参与计算。
        if use_edge_weight and edge_weight is not None:
            if edge_weight.dim() == 1:
                ew = edge_weight.unsqueeze(0).expand(B, -1)
            else:
                ew = edge_weight
            scores = scores + torch.log(ew.clamp_min(1e-8)).unsqueeze(1)

        attn = _edge_softmax(scores, dst, N)
        attn = self.dropout(attn)

        messages = attn.unsqueeze(-1) * v_src  # [B,H,E,D]
        context = torch.zeros(
            B, self.n_heads, N, self.d_k,
            device=x.device, dtype=x.dtype
        )
        scatter_index = dst.view(1, 1, E, 1).expand(B, self.n_heads, E, self.d_k)
        context.scatter_add_(2, scatter_index, messages)

        context = context.transpose(1, 2).contiguous().view(B, N, self.out_features)
        return self.w_o(context)


class GATNetwork(nn.Module):
    """稀疏图注意力网络，接口支持 edge_index，也兼容传入 dense adjacency。"""

    def __init__(self, node_features: int, hidden_features: int, out_features: int,
                 n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(node_features, hidden_features)

        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_features
            out_dim = out_features if i == n_layers - 1 else hidden_features
            self.gat_layers.append(
                SparseMultiHeadGraphAttention(in_dim, out_dim, n_heads, dropout)
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_features) for _ in range(max(0, n_layers - 1))
        ])
        self.final_norm = nn.LayerNorm(out_features) if n_layers > 0 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, graph, edge_weight=None, use_edge_weight=False) -> torch.Tensor:
        if x.dim() == 4:
            batch_size, seq_len, num_nodes, node_features = x.shape
            x_reshaped = x.reshape(batch_size * seq_len, num_nodes, node_features)
            reshape_back = True
        elif x.dim() == 3:
            x_reshaped = x
            batch_size = seq_len = None
            num_nodes = x.size(1)
            reshape_back = False
        else:
            raise ValueError(f'输入 x 必须为 3/4 维，实际为 {x.dim()} 维')

        # graph 既可为 edge_index [2,E]，也可为 dense adjacency [N,N]
        if graph.dim() == 2 and graph.size(0) == 2 and graph.dtype in (torch.int32, torch.int64):
            edge_index = graph.long()
        else:
            edge_index, dense_edge_weight = dense_adj_to_edges(graph)
            edge_index = edge_index.to(x.device)
            if edge_weight is None:
                edge_weight = dense_edge_weight.to(x.device)

        h = self.dropout(self.activation(self.input_proj(x_reshaped)))

        for i, gat_layer in enumerate(self.gat_layers):
            residual = h
            h = gat_layer(
                h, edge_index, edge_weight=edge_weight,
                use_edge_weight=use_edge_weight
            )
            if i < len(self.gat_layers) - 1:
                h = self.activation(h)
                # 中间层维度一致时使用残差
                if h.shape[-1] == residual.shape[-1]:
                    h = self.layer_norms[i](h + residual)
                else:
                    h = self.layer_norms[i](h)
                h = self.dropout(h)

        h = self.final_norm(h)
        if reshape_back:
            return h.reshape(batch_size, seq_len, num_nodes, -1)
        return h


class SparseGraphConvLayer(nn.Module):
    """Edge-only GCN 聚合层：O(E) 而非 dense O(N^2)。"""

    def __init__(self, in_feats, out_feats, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index, edge_weight):
        # x: [B,N,F], edge_index: [2,E], edge_weight: [E]
        h = self.linear(x)
        dst, src = edge_index[0], edge_index[1]
        msg = h[:, src, :] * edge_weight.view(1, -1, 1).to(h.dtype)
        out = torch.zeros_like(h)
        idx = dst.view(1, -1, 1).expand(h.size(0), -1, h.size(-1))
        out.scatter_add_(1, idx, msg)
        return F.relu(out)


class SparseGCNModule(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, n_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        if n_layers == 1:
            layers.append(SparseGraphConvLayer(in_feats, out_feats))
        else:
            layers.append(SparseGraphConvLayer(in_feats, hidden_feats))
            for _ in range(n_layers - 2):
                layers.append(SparseGraphConvLayer(hidden_feats, hidden_feats))
            layers.append(SparseGraphConvLayer(hidden_feats, out_feats))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_feats)

    def forward(self, x, edge_index, edge_weight):
        h = x
        for layer in self.layers:
            h = self.dropout(layer(h, edge_index, edge_weight))
        return self.norm(h)


# ---------- 辅助：邻接矩阵归一化 ----------
# 修复邻接矩阵归一化函数
def normalize_adj(A):
    """对称归一化 A_hat = D^-1/2 (A+I) D^-1/2。

    第一阶段优化：使用逐元素广播替代显式 diag + 矩阵乘法。
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32)
    A = A.float()
    N = A.shape[0]
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    A_hat = A + I
    deg = A_hat.sum(dim=1).clamp_min(1e-8)
    deg_inv_sqrt = deg.rsqrt()
    A_norm = deg_inv_sqrt[:, None] * A_hat * deg_inv_sqrt[None, :]
    A_norm = (A_norm + A_norm.t()) * 0.5
    return torch.clamp(A_norm, 0, 1)


# ---------- 轻量 Graph Convolution 层 ----------
class GraphConvLayer(nn.Module):
    """
    简洁的图卷积层，避免全图 attention 的 B*N*N*F 中间量。
    forward:
      X: [B, N, in_feats]
      A_norm: [N,N] 或 [B,N,N]
    返回:
      out: [B, N, out_feats]
    """

    def __init__(self, in_feats, out_feats, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, X, A_norm):
        # X: [B,N,F]
        H = self.linear(X)  # [B,N,out]
        if A_norm.dim() == 2:
            # 使用 einsum，内存友好
            out = torch.einsum('ij,bjf->bif', A_norm, H)  # [B,N,out]
        else:
            out = torch.bmm(A_norm, H)  # [B,N,out]
        return F.relu(out)


# ---------- 多层轻量 GCN 模块 (用于 geo 与 hist 两张图) ----------
class LightweightGCNModule(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, n_layers=2, dropout=0.1):
        """
        in_feats: 输入特征维度
        hidden_feats: 中间隐藏维度
        out_feats: 输出特征维度
        n_layers: 层数 (>=1)
        """
        super().__init__()
        layers = []
        if n_layers == 1:
            layers.append(GraphConvLayer(in_feats, out_feats))
        else:
            layers.append(GraphConvLayer(in_feats, hidden_feats))
            for _ in range(n_layers - 2):
                layers.append(GraphConvLayer(hidden_feats, hidden_feats))
            layers.append(GraphConvLayer(hidden_feats, out_feats))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_feats)

    def forward(self, x, A_norm):
        """
        x: [B, N, in_feats]
        A_norm: [N,N] or [B,N,N]
        返回: [B, N, out_feats]
        """
        h = x
        for layer in self.layers:
            h = layer(h, A_norm)
            h = self.dropout(h)
        h = self.norm(h)
        return h


# ---------- 用 GCN 替代原先 GAT 的 Spatial Module ----------
class LightweightATGCN_SpatialModule(nn.Module):
    """
    原先使用 GAT 的 Spatial 模块替换为轻量化 GCN 实现，接口保持兼容。
    forward 接收:
      x: [B, seq_len, N, feat]
      adj_geo: [N,N] 或 [B,N,N] (未归一化或归一化均可，但推荐传入归一化后的 A_norm)
      adj_hist: [N,N] 或 [B,N,N]
    返回:
      fused: [B, seq_len, N, out_feats]
    """

    def __init__(self, num_nodes, node_features, gcn_hidden=64, out_feats=64, gcn_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.geo_gcn = LightweightGCNModule(in_feats=node_features, hidden_feats=gcn_hidden, out_feats=out_feats,
                                            n_layers=gcn_layers, dropout=dropout)
        self.hist_gcn = LightweightGCNModule(in_feats=node_features, hidden_feats=gcn_hidden, out_feats=out_feats,
                                             n_layers=gcn_layers, dropout=dropout)
        # 融合网络，输出 alpha 权重 -> sigmoid( ) ∈ (0,1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_feats * 2, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, 1),
            nn.Sigmoid()
        )
        self.final_norm = nn.LayerNorm(out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_geo, adj_hist):
        """
        x: [B, seq_len, N, feat]
        adj_geo / adj_hist: [N,N] 或 [B,N,N]
        """
        B, seq_len, N, feat = x.shape
        # 将时间维和 batch 合并，逐时间步用图卷积处理 -> batch-friendly
        x_rs = x.reshape(B * seq_len, N, feat)  # [B*seq, N, feat]

        # 如果传入的 adj 是 numpy，需要转成 tensor（建议在外部一次性归一化并转成 tensor）
        # geo_out / hist_out: [B*seq, N, out_feats]
        geo_out = self.geo_gcn(x_rs, adj_geo)
        hist_out = self.hist_gcn(x_rs, adj_hist)

        # 融合
        concat = torch.cat([geo_out, hist_out], dim=-1)  # [B*seq, N, out*2]
        alpha = self.fusion_mlp(concat)  # [B*seq, N, 1]
        fused = alpha * geo_out + (1 - alpha) * hist_out  # [B*seq, N, out_feats]
        fused = self.final_norm(fused)
        fused = self.dropout(fused)
        fused = fused.reshape(B, seq_len, N, -1)  # [B, seq_len, N, out_feats]
        return fused


# === 4) 修正 BiLSTMEnhancedInformer: 接收 pred_len, 维度对齐 ===
class BiLSTMEnhancedInformer(nn.Module):
    """BiLSTM增强的Informer时间特征提取模块 (修复版)"""

    def __init__(self, d_model, pred_len, n_layers=2, n_heads=8, d_ff=2048,
                 lstm_hidden=256, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = InformerEncoderWithBiLSTM(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers
        )
        self.decoder = LSTMDecoder(
            d_model=d_model, pred_len=pred_len, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, lstm_hidden=lstm_hidden, lstm_layers=max(1, lstm_layers // 2)
        )

    def forward(self, x_enc, x_dec):
        # x_enc: [B, seq_len, d_model]
        # x_dec: [B, label_len + pred_len, d_model] (typical informer usage)
        x_enc = self.pos_encoding(x_enc)
        x_dec = self.pos_encoding(x_dec)
        enc_out = self.encoder(x_enc)  # [B, seq_len, d_model]
        dec_out = self.decoder(x_dec, enc_out)  # [B, pred_len]
        return dec_out


class DynamicGraphLearner(nn.Module):
    """Edge-only 动态图结构学习器。

    原实现先展开 [B,N,N,D] 节点对，再在最后乘 static_adj 掩码。
    这里直接在 static graph 的 E 条实际边上预测权重，避免 N^2 pair tensor。
    """

    def __init__(self, num_nodes: int, node_features: int, hidden_dim: int = 128,
                 temp_emb_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.temp_emb_dim = temp_emb_dim

        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.temporal_encoder = nn.GRU(
            input_size=node_features,
            hidden_size=temp_emb_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + temp_emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_dim // 2), 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        for name, param in self.temporal_encoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _temporal_embedding(self, x: torch.Tensor, seq_len: int):
        B, T, N, Fdim = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, Fdim)
        temporal_out, _ = self.temporal_encoder(x_reshaped)
        temporal_features = temporal_out[:, -min(T, seq_len):, :]

        # T 通常只有 12，此处保留原始时序自注意力逻辑
        temporal_attention = torch.softmax(
            torch.matmul(temporal_features, temporal_features.transpose(1, 2)),
            dim=-1
        )
        weighted_temporal = torch.matmul(temporal_attention, temporal_features)
        temporal_emb = weighted_temporal.mean(dim=1).reshape(B, N, -1)
        return temporal_emb

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                static_edge_weight: torch.Tensor, seq_len: int = 12) -> torch.Tensor:
        """
        x: [B,T,N,F]
        edge_index: [2,E]，0=dst(i), 1=src(j)
        static_edge_weight: [E]
        return dynamic_edge_weight: [B,E]，已按目标节点行归一化
        """
        B, T, N, _ = x.shape
        dst, src = edge_index[0], edge_index[1]
        E = dst.numel()

        temporal_emb = self._temporal_embedding(x, seq_len)
        node_emb = self.node_embedding.weight  # [N,D]

        # 只 gather 实际 E 条边，不构造 N×N 节点对
        e_dst = node_emb[dst].unsqueeze(0).expand(B, E, -1)
        e_src = node_emb[src].unsqueeze(0).expand(B, E, -1)
        t_dst = temporal_emb[:, dst, :]
        t_src = temporal_emb[:, src, :]

        pair = torch.cat([e_dst, e_src, t_dst, t_src], dim=-1)
        pair_rev = torch.cat([e_src, e_dst, t_src, t_dst], dim=-1)

        raw = self.edge_predictor(pair).squeeze(-1)
        raw_rev = self.edge_predictor(pair_rev).squeeze(-1)
        # 等价于原代码“对称化”思想，且不需要 dense matrix
        raw_sym = 0.5 * (raw + raw_rev)

        weighted = raw_sym * static_edge_weight.view(1, E).to(raw_sym.dtype)

        # 按 adjacency 行（目标节点）归一化
        row_sum = torch.zeros(B, N, device=x.device, dtype=weighted.dtype)
        row_sum.scatter_add_(1, dst.view(1, E).expand(B, E), weighted)
        dynamic_edge_weight = weighted / row_sum.gather(
            1, dst.view(1, E).expand(B, E)
        ).clamp_min(1e-8)

        return dynamic_edge_weight


class MultiScaleTemporalGCN(nn.Module):
    """修复的多尺度时序图卷积网络"""

    def __init__(self, in_feats: int, out_feats: int,
                 temporal_scales: list = [3, 6, 12], dropout: float = 0.1):
        super().__init__()
        self.temporal_scales = temporal_scales
        self.out_feats = out_feats

        # 多尺度时序卷积
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(in_feats, out_feats, kernel_size=scale, padding=scale // 2)
            for scale in temporal_scales
        ])

        self.scale_attention = nn.Sequential(
            nn.Linear(out_feats * len(temporal_scales), out_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feats, len(temporal_scales)),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, T, F = x.shape

        # 重塑为卷积需要的格式: [B*N, F, T]
        x_reshaped = x.permute(0, 1, 3, 2).contiguous().view(B * N, F, T)

        scale_features = []
        for conv in self.temporal_convs:
            # 时序卷积
            conv_out = conv(x_reshaped)  # [B*N, out_feats, T]

            # 聚合时间维度（取均值）
            temporal_feat = conv_out.mean(dim=2)  # [B*N, out_feats]
            temporal_feat = temporal_feat.view(B, N, self.out_feats)  # [B, N, out_feats]
            scale_features.append(temporal_feat)

        # 多尺度融合
        concat = torch.cat(scale_features, dim=-1)  # [B, N, out_feats*num_scales]
        weights = self.scale_attention(concat)  # [B, N, num_scales]

        # 加权求和
        output = torch.zeros(B, N, self.out_feats, device=x.device)
        for i in range(len(self.temporal_scales)):
            output += weights[:, :, i].unsqueeze(-1) * scale_features[i]

        return self.dropout(output)


class DynamicSpatialFusion(nn.Module):
    """四分支动态融合。

    修复原代码在 forward 中临时 new nn.Linear 的问题：该层不会被优化器注册，
    且每次 forward 都随机初始化。现在通过 node_adapter 在 __init__ 中一次性注册。
    """

    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int = 128,
                 num_branches: int = 4, dropout: float = 0.1,
                 node_emb_dim: Optional[int] = None):
        super().__init__()
        self.num_branches = num_branches
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        node_emb_dim = hidden_dim if node_emb_dim is None else node_emb_dim

        self.branch_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_branches)
        ])

        self.node_adapter = (
            nn.Identity() if node_emb_dim == hidden_dim
            else nn.Linear(node_emb_dim, hidden_dim)
        )

        gate_input_dim = hidden_dim * (num_branches + 1)
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_branches),
            nn.Softmax(dim=-1)
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, branch_outputs: list, node_embeddings: torch.Tensor):
        B, N, _ = branch_outputs[0].shape
        adapted_features = [
            adapter(branch_out)
            for adapter, branch_out in zip(self.branch_adapters, branch_outputs)
        ]
        concat_features = torch.cat(adapted_features, dim=-1)

        if node_embeddings.dim() == 2:
            node_emb = node_embeddings.unsqueeze(0).expand(B, -1, -1)
        else:
            node_emb = node_embeddings
        node_emb = self.node_adapter(node_emb)

        gate_input = torch.cat([concat_features, node_emb], dim=-1)
        fusion_weights = self.fusion_gate(gate_input)

        fused_features = sum(
            weight.unsqueeze(-1) * feat
            for weight, feat in zip(fusion_weights.unbind(-1), adapted_features)
        )
        output = self.output_projection(fused_features)
        output = self.norm(output + adapted_features[0])
        return output, fusion_weights


class EnhancedDynamicATGCN_SpatialModule(nn.Module):
    """第一阶段稀疏化的动态多分支空间特征模块。

    四分支语义保持：
      1) Geo GCN
      2) Historical GAT
      3) Dynamic GAT
      4) Multi-scale Temporal Conv

    关键优化：
      - 固定图只在首次 forward 时 dense→edge 转换并缓存；
      - Geo GCN 使用 edge-only 聚合；
      - Hist/Dynamic GAT 不构造 [B,H,N,N]；
      - DynamicGraphLearner 不构造 [B,N,N,D] pair tensor；
      - 不再扩展 dynamic_adj 到 [B*T,N,N]。
    """

    def __init__(self, num_nodes: int, node_features: int,
                 gcn_hidden: int = 128, out_feats: int = 128,
                 gcn_layers: int = 3, dropout: float = 0.1,
                 dynamic_hidden: int = 128, temp_emb_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features

        self.dynamic_graph_learner = DynamicGraphLearner(
            num_nodes=num_nodes,
            node_features=node_features,
            hidden_dim=dynamic_hidden,
            temp_emb_dim=temp_emb_dim,
            dropout=dropout
        )

        self.static_geo_gcn = SparseGCNModule(
            in_feats=node_features,
            hidden_feats=gcn_hidden,
            out_feats=out_feats,
            n_layers=gcn_layers,
            dropout=dropout
        )
        self.static_hist_gat = GATNetwork(
            node_features=node_features,
            hidden_features=gcn_hidden,
            out_features=out_feats,
            n_layers=gcn_layers,
            n_heads=8,
            dropout=dropout
        )
        self.dynamic_gat = GATNetwork(
            node_features=node_features,
            hidden_features=gcn_hidden,
            out_features=out_feats,
            n_layers=gcn_layers,
            n_heads=8,
            dropout=dropout
        )
        self.temporal_gcn = MultiScaleTemporalGCN(
            in_feats=node_features,
            out_feats=out_feats,
            temporal_scales=[3, 6, 12],
            dropout=dropout
        )

        self.fusion_module = DynamicSpatialFusion(
            num_nodes=num_nodes,
            input_dim=out_feats,
            hidden_dim=out_feats,
            num_branches=4,
            dropout=dropout,
            node_emb_dim=dynamic_hidden
        )
        self.output_adapter = nn.Sequential(
            nn.Linear(out_feats, out_feats * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feats * 2, out_feats)
        )
        self.final_norm = nn.LayerNorm(out_feats)

        # 缓存固定图的 edge_index/edge_weight，不写入 checkpoint
        self.register_buffer('_geo_edge_index', torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer('_geo_edge_weight', torch.empty(0), persistent=False)
        self.register_buffer('_hist_edge_index', torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer('_hist_edge_weight', torch.empty(0), persistent=False)
        self._graph_signature = None

    def reset_graph_cache(self):
        self._geo_edge_index = torch.empty((2, 0), dtype=torch.long, device=self._geo_edge_index.device)
        self._geo_edge_weight = torch.empty(0, device=self._geo_edge_weight.device)
        self._hist_edge_index = torch.empty((2, 0), dtype=torch.long, device=self._hist_edge_index.device)
        self._hist_edge_weight = torch.empty(0, device=self._hist_edge_weight.device)
        self._graph_signature = None

    @torch.no_grad()
    def _ensure_graph_cache(self, adj_geo, adj_hist):
        sig = (
            adj_geo.data_ptr(), getattr(adj_geo, '_version', 0),
            adj_hist.data_ptr(), getattr(adj_hist, '_version', 0),
            str(adj_geo.device), str(adj_hist.device)
        )
        if self._graph_signature == sig and self._geo_edge_index.numel() > 0:
            return

        geo_idx, geo_w = dense_adj_to_edges(adj_geo)
        hist_idx, hist_w = dense_adj_to_edges(adj_hist)
        self._geo_edge_index = geo_idx.to(adj_geo.device)
        self._geo_edge_weight = geo_w.to(adj_geo.device)
        self._hist_edge_index = hist_idx.to(adj_hist.device)
        self._hist_edge_weight = hist_w.to(adj_hist.device)
        self._graph_signature = sig

        print(
            f'[GraphCache] geo edges={self._geo_edge_index.size(1):,}, '
            f'hist edges={self._hist_edge_index.size(1):,}, '
            f'N={self.num_nodes}'
        )

    def forward(self, x: torch.Tensor, adj_geo: torch.Tensor,
                adj_hist: torch.Tensor) -> torch.Tensor:
        B, seq_len, N, feat = x.shape
        self._ensure_graph_cache(adj_geo, adj_hist)

        x_reshaped = x.reshape(B * seq_len, N, feat)

        # 1) Edge-only 动态边权：[B,E_geo]
        dynamic_edge_weight = self.dynamic_graph_learner(
            x,
            self._geo_edge_index,
            self._geo_edge_weight,
            seq_len=min(seq_len, 12)
        )

        # 2.1) Geo GCN：edge-only
        geo_features = self.static_geo_gcn(
            x_reshaped, self._geo_edge_index, self._geo_edge_weight
        )

        # 2.2) Historical GAT：保持原代码“邻接作为 mask”的语义
        hist_features = self.static_hist_gat(
            x_reshaped, self._hist_edge_index,
            edge_weight=None, use_edge_weight=False
        )

        # 2.3) Dynamic GAT：每个原始 batch 的动态边权复用到该样本的全部时间步
        dynamic_weight_bt = dynamic_edge_weight.repeat_interleave(seq_len, dim=0)
        dynamic_features = self.dynamic_gat(
            x_reshaped, self._geo_edge_index,
            edge_weight=dynamic_weight_bt,
            use_edge_weight=True
        )

        # 2.4) Multi-scale temporal branch
        x_temporal = x.permute(0, 2, 1, 3)
        temporal_features = self.temporal_gcn(x_temporal, adj_geo)
        temporal_features = temporal_features.unsqueeze(1).expand(B, seq_len, N, -1)
        temporal_features = temporal_features.reshape(B * seq_len, N, -1)

        branch_outputs = [
            geo_features, hist_features, dynamic_features, temporal_features
        ]
        node_embeddings = self.dynamic_graph_learner.node_embedding.weight

        # 融合保持分块，避免峰值显存过高
        fusion_chunk = 32
        fused_features_list = []
        for i in range(0, B * seq_len, fusion_chunk):
            end_idx = min(i + fusion_chunk, B * seq_len)
            batch_branches = [branch[i:end_idx] for branch in branch_outputs]
            fused_batch, _ = self.fusion_module(batch_branches, node_embeddings)
            fused_features_list.append(fused_batch)

        fused_features = torch.cat(fused_features_list, dim=0)
        output = self.final_norm(self.output_adapter(fused_features))
        return output.reshape(B, seq_len, N, -1)


# 集成到主模型中的修改
class DynamicLightweightATGCN_InformerWithBiLSTM(nn.Module):
    """MBSTGFormer 第一阶段代码级优化版。

    本阶段不缩小 d_model / hidden size，不做知识蒸馏与结构剪枝。
    重点是消除无效参数、消除 N^2 中间张量并稀疏化图计算。
    """

    def __init__(self, num_nodes, node_features, enc_in, dec_in, c_out=1,
                 seq_len=24, label_len=12, pred_len=1, d_model=128,
                 n_heads=4, e_layers=1, d_layers=1, d_ff=256,
                 dropout=0.1, lstm_hidden=128, lstm_layers=1,
                 gcn_hidden=64, gcn_layers=2, dynamic_hidden=64,
                 temp_emb_dim=32, spatial_out=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.st_embed = SpatioTemporalEmbed(num_nodes, spatial_out)
        self.ts_decomposition = TrendSeasonalDecomposition(num_nodes)

        self.trend_informer_enc = InformerEncoderWithBiLSTM(
            d_model=spatial_out, n_layers=e_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers
        )
        self.seasonal_informer_enc = InformerEncoderWithBiLSTM(
            d_model=spatial_out, n_layers=e_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers
        )

        self.ff_t = nn.Sequential(
            nn.Linear(spatial_out, spatial_out), nn.ReLU(),
            nn.Linear(spatial_out, spatial_out)
        )
        self.ff_s = nn.Sequential(
            nn.Linear(spatial_out, spatial_out), nn.ReLU(),
            nn.Linear(spatial_out, spatial_out)
        )

        self.spatial_module = EnhancedDynamicATGCN_SpatialModule(
            num_nodes=num_nodes,
            node_features=node_features,
            gcn_hidden=gcn_hidden,
            out_feats=spatial_out,
            gcn_layers=gcn_layers,
            dropout=dropout,
            dynamic_hidden=dynamic_hidden,
            temp_emb_dim=temp_emb_dim
        )

        self.decoder = LSTMDecoder(
            d_model=d_model, pred_len=pred_len,
            n_layers=d_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=max(1, lstm_layers // 2)
        )

        self.input_proj = nn.Linear(node_features, d_model)
        self.trend_proj = nn.Linear(spatial_out, d_model)

        # 已删除原代码中从未被 forward 使用的：
        #   self.feature_adapter
        #   self.seasonal_proj
        print(f'第一阶段优化模型参数量: {sum(p.numel() for p in self.parameters()):,}')

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, adj_geo, adj_hist):
        B, seq_len, N, feat = x_enc.shape
        if N != self.num_nodes:
            raise ValueError(f'节点数不匹配：模型={self.num_nodes}, 输入={N}')

        # 1) 稀疏动态多分支空间建模
        spatial_feat = self.spatial_module(x_enc, adj_geo, adj_hist)

        # 2) 使用真实地理图构造时空嵌入
        M = self.st_embed(x_mark_enc, adj_geo)

        # 3) 趋势-季节分解（删除原先未使用的 result）
        trend, seasonal = self.ts_decomposition(spatial_feat, M)
        trend_ff = self.ff_t(trend) + trend
        seasonal_ff = self.ff_s(seasonal) + seasonal

        trend_rs = trend_ff.permute(0, 2, 1, 3).reshape(B * N, seq_len, -1)
        seasonal_rs = seasonal_ff.permute(0, 2, 1, 3).reshape(B * N, seq_len, -1)

        trend_enc = self.trend_informer_enc(trend_rs)
        seasonal_enc = self.seasonal_informer_enc(seasonal_rs)

        ts_encoded = self.trend_proj(trend_enc + seasonal_enc)

        # 4) Decoder 输入
        dec_proj = self.input_proj(x_dec)
        dec_input = dec_proj.permute(0, 2, 1, 3).reshape(B * N, dec_proj.size(1), -1)

        # 修复原代码参数顺序：LSTMDecoder.forward(x, enc_output)
        # x 应为 decoder input，enc_output 应为 encoder memory。
        output = self.decoder(dec_input, ts_encoded)

        return output.reshape(B, N, self.pred_len).permute(0, 2, 1)


# 辅助函数：动态图可视化
def visualize_dynamic_graph(dynamic_adj, static_adj, node_names=None, timestep=0):
    """可视化动态图结构变化"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 静态图
    sns.heatmap(static_adj.cpu().numpy(), ax=ax1, cmap='viridis')
    ax1.set_title('Static Adjacency Matrix')

    # 动态图
    dynamic_np = dynamic_adj[timestep].cpu().detach().numpy()
    sns.heatmap(dynamic_np, ax=ax2, cmap='viridis')
    ax2.set_title(f'Dynamic Adjacency Matrix (t={timestep})')

    # 差异图
    diff = dynamic_np - static_adj.cpu().numpy()
    sns.heatmap(diff, ax=ax3, cmap='RdBu_r', center=0)
    ax3.set_title('Difference (Dynamic - Static)')

    plt.tight_layout()
    plt.show()

def interp_5to1(x):
    """
    x: np.ndarray [T, N, F]  原始 5 min 数据
    return: 同 shape，但峰值被重建
    """
    T, N, F = x.shape
    old_t = np.arange(T)                    # 0,1,2,...,T-1
    new_t = np.arange(0, T-0.9, 0.2)        # 0,0.2,0.4,... → 5×密度
    out = np.empty_like(x, dtype=np.float32)
    for n in range(N):
        for f in range(F):
            y = x[:, n, f]
            # 线性插值
            y_interp = interp1d(old_t, y, kind='linear', axis=0)(new_t)
            # 在原 5 min 位置抽回
            out[:, n, f] = y_interp[::5]
    return out
    
class PEMSDatasetProcessor:
    """PEMS04数据集处理器"""

    def __init__(self, data_path: str, seq_len: int = 24, label_len: int = 12, pred_len: int = 1,
                 base_start_time: str = '2018-01-01'):
        self.data_path = data_path
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scaler = MinMaxScaler()
        self.freq = '5min'  # 固定频率
        self.base_start_time = base_start_time

    def load_adjacency_matrix(self, distance_file: Optional[str] = None, threshold: float = 0.1) -> torch.Tensor:
        """从距离文件加载邻接矩阵"""
        if distance_file is not None:
            try:
                df = pd.read_csv(distance_file)

                if len(df.columns) == 3 and 'from' in df.columns and 'to' in df.columns and 'cost' in df.columns:
                    print("检测到标准表头格式")
                else:
                    df = pd.read_csv(distance_file, names=['from', 'to', 'cost'])
                    print("使用自定义列名格式")

                print(f"成功读取边列表数据，行数: {len(df)}")

                df['from'] = pd.to_numeric(df['from'], errors='coerce')
                df['to'] = pd.to_numeric(df['to'], errors='coerce')
                df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

                df = df.dropna()

                max_node = max(df['from'].max(), df['to'].max()) + 1
                num_nodes = max(int(max_node), 307)

                print(f"检测到最大节点ID: {max_node}, 使用节点数: {num_nodes}")

                dist_matrix = np.full((num_nodes, num_nodes), 1e8)
                np.fill_diagonal(dist_matrix, 0)

                for _, row in df.iterrows():
                    i, j, cost = int(row['from']), int(row['to']), float(row['cost'])
                    dist_matrix[i][j] = cost
                    dist_matrix[j][i] = cost

                print(f"构建距离矩阵完成，形状: {dist_matrix.shape}")

                similarity = 1.0 / (dist_matrix + 1e-8)
                similarity[similarity > threshold] = threshold

                sim_min = similarity.min()
                sim_max = similarity.max()
                if sim_max - sim_min > 1e-8:
                    similarity = (similarity - sim_min) / (sim_max - sim_min)
                else:
                    similarity = np.zeros_like(similarity)

                np.fill_diagonal(similarity, 1.0)

                print(f"邻接矩阵构建完成，形状: {similarity.shape}")
                print(f"相似度范围: [{similarity.min():.4f}, {similarity.max():.4f}]")

                return torch.FloatTensor(similarity)

            except Exception as e:
                print(f"无法加载邻接矩阵 {distance_file}: {e}")
                import traceback
                traceback.print_exc()

        print("使用默认单位矩阵作为邻接矩阵")
        num_nodes = 307
        return torch.eye(num_nodes)

    def compute_historical_similarity(self, data: torch.Tensor) -> torch.Tensor:
        """计算历史相似度矩阵"""
        if isinstance(data, torch.Tensor):
            data_np = data.numpy()
        else:
            data_np = data

        # 提取流量特征（第一个特征）
        flow_data = data_np[..., 0]  # [num_timesteps, num_nodes]

        # 计算皮尔逊相关系数
        correlation_matrix = np.corrcoef(flow_data.T)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        # 转换为相似度矩阵
        similarity_matrix = (correlation_matrix + 1) / 2
        np.fill_diagonal(similarity_matrix, 1.0)

        # 应用阈值
        similarity_matrix[similarity_matrix < 0.3] = 0

        print(f"历史相似度矩阵计算完成，形状: {similarity_matrix.shape}")
        print(f"相似度范围: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

        return torch.FloatTensor(similarity_matrix)

    def load_pems_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
        data = np.load(self.data_path)
        flow_data = data['data'] if 'data' in data else data[data.files[0]]
        flow_data = interp_5to1(flow_data)
        # 分割数据（与之前相同）
        total_len = flow_data.shape[0]
        train_ratio, val_ratio = 0.7, 0.2
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))

        # 原始数据分割
        train_data_original = flow_data[:train_end]
        val_data_original = flow_data[train_end:val_end]
        test_data_original = flow_data[val_end:]

        # 归一化（与之前相同）
        self.scaler = MinMaxScaler()
        train_2d = train_data_original.reshape(-1, flow_data.shape[-1])
        self.scaler.fit(train_2d)

        train_data_normalized = self.scaler.transform(
            train_data_original.reshape(-1, flow_data.shape[-1])
        ).reshape(train_data_original.shape)

        val_data_normalized = self.scaler.transform(
            val_data_original.reshape(-1, flow_data.shape[-1])
        ).reshape(val_data_original.shape)

        test_data_normalized = self.scaler.transform(
            test_data_original.reshape(-1, flow_data.shape[-1])
        ).reshape(test_data_original.shape)

        # 返回原始数据和分割信息，不返回时间特征
        return (torch.FloatTensor(train_data_normalized),
                torch.FloatTensor(val_data_normalized),
                torch.FloatTensor(test_data_normalized),
                torch.FloatTensor(flow_data),  # 原始数据用于参考
                self.scaler,
                train_end, val_end, total_len)  # 返回分割点信息

    def _create_time_features(self, start_idx: int, length: int) -> np.ndarray:
        """
        为指定起始索引和长度的序列独立生成时间特征
        Args:
            start_idx: 序列在原始数据中的起始索引（相对于base_start_time）
            length: 序列长度
        Returns:
            time_features: [length, 4] 时间特征数组
        """
        # 基于起始索引计算绝对时间戳
        start_timestamp = pd.Timestamp(self.base_start_time) + pd.Timedelta(start_idx * 5, 'min')

        # 生成该序列的时间范围
        sequence_timestamps = pd.date_range(
            start=start_timestamp,
            periods=length,
            freq=self.freq
        )

        features = []
        for ts in sequence_timestamps:
            hour = ts.hour / 23.0
            dayofweek = ts.dayofweek / 6.0
            month = (ts.month - 1) / 11.0
            is_weekend = 1.0 if ts.dayofweek >= 5 else 0.0
            features.append([hour, dayofweek, month, is_weekend])

        return np.array(features)

    def create_dataset(self, data_split, split_type='train', start_idx=0):
        """
        安全的数据集创建方法 - 每个序列独立生成时间特征
        Args:
            data_split: 数据切片（训练/验证/测试集）
            split_type: 数据集类型
            start_idx: 该数据集在原始数据中的起始索引
        """
        try:
            if isinstance(data_split, torch.Tensor):
                data_np = data_split.numpy()
            else:
                data_np = data_split

            # 数据验证
            if np.any(np.isnan(data_np)):
                print(f"警告：{split_type}数据中存在NaN值，进行清理")
                data_np = np.nan_to_num(data_np, nan=0.0)

            sequences = []
            labels = []

            max_start_idx = len(data_np) - self.seq_len - self.pred_len
            if max_start_idx <= 0:
                raise ValueError(f"{split_type}数据长度不足")

            print(f"为{split_type}集创建序列，数据长度: {len(data_np)}")

            for i in range(max_start_idx + 1):
                # 计算该序列在全局数据中的绝对起始索引
                global_start_idx = start_idx + i

                # 编码器输入
                enc_start, enc_end = i, i + self.seq_len
                x_enc = data_np[enc_start:enc_end]  # [seq_len, num_nodes, features]

                # 为编码器序列独立生成时间特征
                x_mark_enc = self._create_time_features(
                    global_start_idx, self.seq_len
                )

                # 解码器输入
                dec_start = i + self.seq_len - self.label_len
                global_dec_start = start_idx + dec_start

                x_dec_real = data_np[dec_start:dec_start + self.label_len]
                x_dec_padding = np.zeros((self.pred_len, *x_dec_real.shape[1:]),
                                         dtype=x_dec_real.dtype)
                x_dec = np.concatenate([x_dec_real, x_dec_padding], axis=0)

                # 为解码器序列独立生成时间特征
                x_mark_dec = self._create_time_features(
                    global_dec_start, self.label_len + self.pred_len
                )

                # 标签
                label_start, label_end = i + self.seq_len, i + self.seq_len + self.pred_len
                y = data_np[label_start:label_end, :, 0]  # [pred_len, num_nodes]

                if np.any(np.isnan(x_enc)) or np.any(np.isnan(y)):
                    continue

                sequences.append((x_enc, x_mark_enc, x_dec, x_mark_dec))
                labels.append(y)

            # 转换为tensor
            x_enc_tensor = torch.stack([torch.FloatTensor(s[0]) for s in sequences])
            x_mark_enc_tensor = torch.stack([torch.FloatTensor(s[1]) for s in sequences])
            x_dec_tensor = torch.stack([torch.FloatTensor(s[2]) for s in sequences])
            x_mark_dec_tensor = torch.stack([torch.FloatTensor(s[3]) for s in sequences])
            labels_tensor = torch.stack([torch.FloatTensor(s) for s in labels])

            print(f"{split_type}集生成 {len(sequences)} 个序列")
            return x_enc_tensor, x_mark_enc_tensor, x_dec_tensor, x_mark_dec_tensor, labels_tensor

        except Exception as e:
            print(f"创建{split_type}数据集时出错: {e}")
            return None


# 训练函数
def train_graph_informer(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    """训练GraphInformer模型"""
    model.train()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for batch_idx, (x_enc, x_mark_enc, x_dec, x_mark_dec, y) in enumerate(train_loader):
            x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
            x_mark_enc, x_mark_dec = x_mark_enc.to(device), x_mark_dec.to(device)

            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.6f}')

        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for x_enc, x_mark_enc, x_dec, x_mark_dec, y in val_loader:
                x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                x_mark_enc, x_mark_dec = x_mark_enc.to(device), x_mark_dec.to(device)

                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                val_loss = criterion(output, y)
                epoch_val_loss += val_loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')

    return train_losses, val_losses


import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os


class TrafficTrainer:
    """交通流量预测训练器"""

    def __init__(self, model, device, model_save_path='./models'):
        self.model = model
        self.device = device
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)

        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def create_dataloader(self, dataset, batch_size=32, shuffle=True):
        """创建数据加载器"""
        x_enc, x_mark_enc, x_dec, x_mark_dec, y = dataset

        # 确保数据形状正确
        print(f"数据形状 - x_enc: {x_enc.shape}, x_mark_enc: {x_mark_enc.shape}")
        print(f"x_dec: {x_dec.shape}, x_mark_dec: {x_mark_dec.shape}, y: {y.shape}")

        # 创建TensorDataset
        dataset = TensorDataset(x_enc, x_mark_enc, x_dec, x_mark_dec, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def train_epoch(self, train_loader, optimizer, criterion, adj_geo, adj_hist):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        batch_count = 0
        for batch_idx, (x_enc, x_mark_enc, x_dec, x_mark_dec, y) in enumerate(train_loader):
            x_enc = x_enc.to(self.device).float()
            x_dec = x_dec.to(self.device).float()
            y = y.to(self.device).float()
            x_mark_enc = x_mark_enc.to(self.device).float()
            x_mark_dec = x_mark_dec.to(self.device).float()

            optimizer.zero_grad()
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, adj_geo, adj_hist)

            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if batch_idx % 50 == 0:
                print(f'训练批次 {batch_idx}/{len(train_loader)} | 损失: {loss.item():.6f}')

        return epoch_loss / batch_count if batch_count > 0 else 0

    def validate_epoch(self, val_loader, criterion, adj_geo, adj_hist, scaler=None):
        """验证一个epoch"""
        self.model.eval()
        epoch_loss = 0
        batch_count = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, (x_enc, x_mark_enc, x_dec, x_mark_dec, y) in enumerate(val_loader):
                x_enc = x_enc.to(self.device).float()
                x_dec = x_dec.to(self.device).float()
                y = y.to(self.device).float()
                x_mark_enc = x_mark_enc.to(self.device).float()
                x_mark_dec = x_mark_dec.to(self.device).float()

                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, adj_geo, adj_hist)

                loss = criterion(output, y)
                epoch_loss += loss.item()
                batch_count += 1

                all_predictions.append(output.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        val_loss = epoch_loss / batch_count if batch_count > 0 else 0
        metrics = self.calculate_metrics(all_predictions, all_targets, scaler)

        return val_loss, metrics

    def calculate_metrics(self, predictions, targets, scaler=None):
        """修复的指标计算"""
        '''if not predictions or not targets:
            return {}

        pred_array = np.concatenate(predictions, axis=0)
        target_array = np.concatenate(targets, axis=0)

        # 确保形状一致
        pred_array = pred_array.reshape(-1)
        target_array = target_array.reshape(-1)

        # 过滤无效值
        mask = ~(np.isnan(pred_array) | np.isnan(target_array) | np.isinf(pred_array) | np.isinf(target_array))
        pred_clean = pred_array[mask]
        target_clean = target_array[mask]

        if len(pred_clean) == 0:
            return {}

        # 计算基础指标
        mse = mean_squared_error(target_clean, pred_clean)
        mae = mean_absolute_error(target_clean, pred_clean)
        rmse = np.sqrt(mse)

        # 安全的R²计算
        if target_clean.std() > 1e-8:
            r2 = r2_score(target_clean, pred_clean)
        else:
            r2 = 0.0

        # 安全的MAPE计算
        epsilon = 1e-8
        valid_mask = np.abs(target_clean) > epsilon
        if np.any(valid_mask):
            mape = np.mean(np.abs((target_clean[valid_mask] - pred_clean[valid_mask]) / target_clean[valid_mask])) * 100
        else:
            mape = 0.0

        metrics = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MAPE': float(mape)
        }'''
        """计算评估指标（支持反归一化）"""
        if not predictions or not targets:
            return {}

        pred_array = np.concatenate(predictions, axis=0)
        target_array = np.concatenate(targets, axis=0)

        pred_array = pred_array.reshape(-1, 1)
        target_array = target_array.reshape(-1, 1)

        mask = ~(np.isnan(pred_array) | np.isnan(target_array))
        pred_clean = pred_array[mask]
        target_clean = target_array[mask]

        if len(pred_clean) == 0:
            return {}

        mse = mean_squared_error(target_clean, pred_clean)
        mae = mean_absolute_error(target_clean, pred_clean)
        rmse = np.sqrt(mse)

        if target_clean.std() > 0:
            r2 = r2_score(target_clean, pred_clean)
        else:
            r2 = 0

        epsilon = 1e-8
        mape = np.mean(np.abs((target_clean - pred_clean) / (target_clean + epsilon))) * 100

        norm_metrics = {
            'MSE_norm': mse,
            'MAE_norm': mae,
            'RMSE_norm': rmse,
            'R2_norm': r2,
            'MAPE_norm': mape
        }
        denorm_metrics = {}
        if scaler is not None:
            try:
                pred_3d = np.zeros((len(pred_clean), 3))
                pred_3d[:, 0] = pred_clean.flatten()

                target_3d = np.zeros((len(target_clean), 3))
                target_3d[:, 0] = target_clean.flatten()

                pred_denorm = scaler.inverse_transform(pred_3d)
                target_denorm = scaler.inverse_transform(target_3d)

                pred_flow = pred_denorm[:, 0]
                target_flow = target_denorm[:, 0]

                mse_denorm = mean_squared_error(target_flow, pred_flow)
                mae_denorm = mean_absolute_error(target_flow, pred_flow)
                rmse_denorm = np.sqrt(mse_denorm)

                if target_flow.std() > 0:
                    r2_denorm = r2_score(target_flow, pred_flow)
                else:
                    r2_denorm = 0

                mape_denorm = np.mean(np.abs((target_flow - pred_flow) / (target_flow + epsilon))) * 100

                denorm_metrics = {
                    'MSE_denorm': mse_denorm,
                    'MAE_denorm': mae_denorm,
                    'RMSE_denorm': rmse_denorm,
                    'R2_denorm': r2_denorm,
                    'MAPE_denorm': mape_denorm
                }
            except Exception as e:
                print(f"反归一化失败: {e}")

        return {**norm_metrics, **denorm_metrics}

    def train(self, train_dataset, val_dataset, epochs=100, batch_size=32,
              learning_rate=1e-4, patience=10, adj_geo=None, adj_hist=None, scaler=None):
        """训练模型"""
        print("开始训练模型...")
        adj_geo = adj_geo.to(self.device)
        adj_hist = adj_hist.to(self.device)
        train_loader = self.create_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = self.create_dataloader(val_dataset, batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        criterion = nn.MSELoss()

        early_stopping_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader, optimizer, criterion, adj_geo, adj_hist)

            val_loss, val_metrics = self.validate_epoch(val_loader, criterion, adj_geo, adj_hist, scaler)

            scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            '''if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(f'best_model_epoch_{epoch + 1}.pth')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1'''

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_model.pth')

            epoch_time = time.time() - start_time
            print(f'Epoch {epoch + 1}/{epochs} | 时间: {epoch_time:.2f}s')
            print(f'训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}')

            if val_metrics:
                print(f"归一化指标 - RMSE: {val_metrics.get('RMSE_norm', 0):.4f}, "
                      f"MAE: {val_metrics.get('MAE_norm', 0):.4f}, "
                      f"R²: {val_metrics.get('R2_norm', 0):.4f}")

                if 'RMSE_denorm' in val_metrics:
                    print(f"反归一化指标 - RMSE: {val_metrics['RMSE_denorm']:.2f}, "
                          f"MAE: {val_metrics['MAE_denorm']:.2f}, "
                          f"R²: {val_metrics['R2_denorm']:.4f}, "
                          f"MAPE: {val_metrics['MAPE_denorm']:.2f}%")

            print('-' * 60)

            if early_stopping_counter >= patience and epoch > 70:
                print(f"早停触发！在 epoch {epoch + 1} 停止训练")
                break

        self.save_model('final_model.pth')
        print("训练完成！")

        return self.train_losses, self.val_losses

    def test(self, test_dataset, batch_size=32, adj_geo=None, adj_hist=None, scaler=None):
        """测试模型"""
        print("开始测试模型...")
        adj_geo = adj_geo.to(self.device)
        adj_hist = adj_hist.to(self.device)
        test_loader = self.create_dataloader(test_dataset, batch_size, shuffle=False)
        criterion = nn.MSELoss()

        test_loss, test_metrics = self.validate_epoch(test_loader, criterion, adj_geo, adj_hist, scaler)

        print(f"测试损失: {test_loss:.6f}")
        if test_metrics:
            print(f"测试指标:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")

        return test_loss, test_metrics

    def save_model(self, filename):
        """保存模型"""
        save_path = os.path.join(self.model_save_path, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, save_path)
        print(f"模型已保存到: {save_path}")

    def load_model(self, filename):
        """加载模型"""
        load_path = os.path.join(self.model_save_path, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"模型已从 {load_path} 加载")

    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('训练和验证损失')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        if len(self.train_losses) > 50:
            plt.plot(range(len(self.train_losses) - 50, len(self.train_losses)),
                     self.train_losses[-50:], label='训练损失')
            plt.plot(range(len(self.val_losses) - 50, len(self.val_losses)),
                     self.val_losses[-50:], label='验证损失')
        else:
            plt.plot(self.train_losses, label='训练损失')
            plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('训练和验证损失（最近50个epoch）')

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_history.png'))
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 模型参数
    num_nodes = 307
    node_features = 3
    seq_len = 12
    label_len = 12
    pred_len = 12

    # 大幅简化的模型参数
    model = DynamicLightweightATGCN_InformerWithBiLSTM(
        num_nodes=307,
        node_features=3,
        enc_in=3,
        dec_in=3,
        c_out=1,
        seq_len=12,  # 减少序列长度
        label_len=12,  # 减少标签长度
        pred_len=12,
        d_model=256,  # 大幅降低
        n_heads=4,  # 减少头数
        e_layers=1,  # 减少编码器层数
        d_layers=1,  # 减少解码器层数
        d_ff=128,  # 减小前馈网络维度
        lstm_hidden=128,
        lstm_layers=1,
        gcn_hidden=128,
        gcn_layers=2,
        dynamic_hidden=128,
        spatial_out=128,
        temp_emb_dim=64,dropout=0.15
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 数据处理器
    processor = PEMSDatasetProcessor(
        data_path=r"D:\Pycharm\PythonProject\data\data\PEMS04\pems04.npz",  # 替换为实际路径
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len
    )

    # 加载数据
    # data, time_features, original_data, scaler = processor.load_pems_data()
    train_data, val_data, test_data, original_data, scaler, train_end, val_end, total_len = processor.load_pems_data()
    # 计算各数据集的起始索引
    train_start_idx = 0
    val_start_idx = train_end
    test_start_idx = val_end

    # 加载邻接矩阵
    # 先归一化（只做一次），然后把归一化结果发到 device 并使用它们
    adj_geo = processor.load_adjacency_matrix(r"D:\Pycharm\PythonProject\data\data\PEMS04\distance4.csv")
    # 只使用训练数据计算历史相似度

    adj_hist = processor.compute_historical_similarity(original_data[:train_end])  # 仅使用训练段，避免数据泄露
    # train_end = int(data.shape[0] * 0.7)  # 使用归一化后的数据
    # train_data_normalized = data[:train_end]
    # adj_hist = processor.compute_historical_similarity(train_data_normalized)

    # 归一化（normalize_adj 支持 numpy 或 tensor）
    adj_geo_norm = normalize_adj(adj_geo)
    adj_hist_norm = normalize_adj(adj_hist)

    # 把归一化后的邻接矩阵放到 device 上（并使用这个 norm 版本）
    adj_geo = adj_geo_norm.to(device)
    adj_hist = adj_hist_norm.to(device)

    # 或者直接计算历史相似度矩阵
    # adj_hist = processor.compute_historical_similarity(original_data)

    # 创建数据集

    # 创建数据集（分别传入各数据集）
    train_dataset = processor.create_dataset(train_data, 'train', train_start_idx)
    val_dataset = processor.create_dataset(val_data, 'val', val_start_idx)
    test_dataset = processor.create_dataset(test_data, 'test', test_start_idx)
    # 创建训练器
    trainer = TrafficTrainer(model, device)

    # 训练参数
    epochs = 100
    batch_size = 8 # 根据GPU内存调整
    learning_rate = 1e-4
    patience = 15

    # 开始训练
    train_losses, val_losses = trainer.train(
        train_dataset, val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        adj_geo=adj_geo,
        adj_hist=adj_hist,
        scaler=scaler
    )

    # 绘制训练历史
    trainer.plot_training_history()

    # 测试模型
    test_loss, test_metrics = trainer.test(
        test_dataset,
        batch_size=batch_size,
        adj_geo=adj_geo,
        adj_hist=adj_hist,
        scaler=scaler
    )

    # 保存结果
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'model_params': sum(p.numel() for p in model.parameters())
    }

    with open(os.path.join(trainer.model_save_path, 'training_results.json'), 'w') as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in results.items()}, f, indent=2)

    print("训练和测试完成！")