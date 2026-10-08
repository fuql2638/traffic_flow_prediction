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
    """Laplacian特征+周期one-hot → 时空嵌入M"""

    def __init__(self, num_nodes, d_model, max_len=5000):
        super().__init__()
        # 1) 空间：Laplacian前k=32特征向量
        laplacian = self._get_laplacian(num_nodes)  # [N,N]
        eig_val, eig_vec = torch.linalg.eigh(laplacian)  # 升序
        self.register_buffer('spatial', eig_vec[:, 1:33])  # [N,32] 去0特征

        # 2) 时间：小时+星期 one-hot → MLP
        self.temp_mlp = nn.Sequential(
            nn.Linear(24 + 7, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # 3) 融合MLP
        self.fusion = nn.Sequential(
            nn.Linear(32 + d_model, d_model),
            nn.Sigmoid()  # 保证0~1，当刀子
        )

    def _get_laplacian(self, N):
        adj = torch.eye(N)  # 这里用单位矩阵示例，你换成adj_geo
        deg = adj.sum(1)
        return torch.eye(N) - torch.diag(deg ** -0.5) @ adj @ torch.diag(deg ** -0.5)

    def forward(self, x_mark_enc):
        """
        x_mark_enc: [B,T,4]  小时/星期/月/周末
        返回 M ∈ [B,T,N,D]  0~1
        """
        B, T, _ = x_mark_enc.shape
        # 时间嵌入
        h = x_mark_enc[..., 0] * 23  # 小时 0~23
        w = x_mark_enc[..., 1] * 6  # 星期 0~6
        h_one = F.one_hot(h.long(), 24).float()
        w_one = F.one_hot(w.long(), 7).float()
        t_emb = self.temp_mlp(torch.cat([h_one, w_one], -1))  # [B,T,D]

        # 空间嵌入
        s_emb = self.spatial.unsqueeze(0).expand(B, -1, -1)  # [B,N,32]

        # 融合
        t_broad = t_emb.unsqueeze(2).expand(-1, -1, s_emb.size(1), -1)  # [B,T,N,D]
        s_broad = s_emb.unsqueeze(1).expand(-1, T, -1, -1)  # [B,T,N,32]
        M = self.fusion(torch.cat([t_broad, s_broad], -1))  # [B,T,N,D] 0~1
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
    """可学习的趋势季节分解模块 - 与model.py中一致"""
    def __init__(self, num_nodes):
        super().__init__()
        # 可学习的融合权重，初始化0.5
        self.vector = nn.Parameter(torch.full((1, 1, num_nodes, 1), 0.5, requires_grad=True))
        
    def forward(self, X, STEmbedding):
        """
        X: [B, T, N, D]
        STEmbedding: [B, T, N, D] 时空嵌入
        返回: [B, T, N, D]
        """
        trend = torch.mul(X, STEmbedding)  # 趋势分量
        seasonal = X - trend  # 季节分量
        
        # 可学习的融合权重
        zero_shape = torch.zeros_like(X)
        vector = zero_shape + self.vector
        
        # 加权融合：vector*trend + (1-vector)*seasonal
        result = vector * trend + (1 - vector) * seasonal
        
        return result, trend, seasonal
        
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
        self.projection = nn.Linear(d_model, pred_len)
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


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.d_k = out_features // n_heads

        self.w_q = nn.Linear(in_features, out_features)
        self.w_k = nn.Linear(in_features, out_features)
        self.w_v = nn.Linear(in_features, out_features)
        self.w_o = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # 实现多头图注意力
        B, N, F = x.shape
        Q = self.w_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if adj is not None:
            adj_mask = adj.unsqueeze(0).unsqueeze(0)  # 扩展维度匹配
            scores = scores.masked_fill(adj_mask == 0, -1e9)

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.w_o(context)

        return output


class GATNetwork(nn.Module):
    """图注意力网络"""

    def __init__(self, node_features: int, hidden_features: int, out_features: int,
                 n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers

        self.input_proj = nn.Linear(node_features, hidden_features)

        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_features if i == 0 else hidden_features
            out_dim = out_features if i == n_layers - 1 else hidden_features
            self.gat_layers.append(
                MultiHeadGraphAttention(in_dim, out_dim, n_heads, dropout)
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_features) for _ in range(n_layers - 1)
        ])
        if n_layers > 0:
            self.final_norm = nn.LayerNorm(out_features)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 检查输入维度并相应处理
        if adj.dim() == 3:
            adj = adj[0]
        adj = adj.squeeze()  # [N, N]
        if x.dim() == 4:
            # 四维输入: [batch_size, seq_len, num_nodes, node_features]
            batch_size, seq_len, num_nodes, node_features = x.shape
            x_reshaped = x.reshape(batch_size * seq_len, num_nodes, node_features)
            reshape_back = True
        elif x.dim() == 3:
            # 三维输入: [batch_size * seq_len, num_nodes, node_features]
            batch_size_seq_len, num_nodes, node_features = x.shape
            x_reshaped = x
            reshape_back = False
        else:
            raise ValueError(f"输入x的维度必须为3或4，但得到的是{x.dim()}维")

        # 输入投影
        h = self.input_proj(x_reshaped)
        h = self.activation(h)
        h = self.dropout(h)

        # GAT层处理
        for i, gat_layer in enumerate(self.gat_layers):
            residual = h
            h = gat_layer(h, adj)

            if i < len(self.gat_layers) - 1:
                h = self.activation(h)
                h = self.layer_norms[i](h + residual)
                h = self.dropout(h)

        if self.n_layers > 0:
            h = self.final_norm(h)

        # 如果需要，重塑回原始维度
        if reshape_back:
            output = h.reshape(batch_size, seq_len, num_nodes, -1)
        else:
            output = h

        return output


# ---------- 辅助：邻接矩阵归一化 ----------
# 修复邻接矩阵归一化函数
def normalize_adj(A):
    """修复的邻接矩阵归一化"""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32)

    A = A.float()
    N = A.shape[0]
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    A_hat = A + I

    # 确保数值稳定性
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt[torch.isnan(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    # 确保对称性和数值范围
    A_norm = (A_norm + A_norm.t()) / 2
    A_norm = torch.clamp(A_norm, 0, 1)

    return A_norm


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
    """动态图结构学习器 - 基于节点嵌入和时序特征"""

    def __init__(self, num_nodes: int, node_features: int, hidden_dim: int = 128,
                 temp_emb_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 节点嵌入学习
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)

        # 时序特征编码器
        self.temporal_encoder = nn.GRU(
            input_size=node_features,
            hidden_size=temp_emb_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 动态边预测器
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + temp_emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.node_embedding.weight)
        for name, param in self.temporal_encoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor,
                seq_len: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 [B, T, N, F]
            static_adj: 静态邻接矩阵 [N, N] 或 [B, N, N]
            seq_len: 用于时序编码的序列长度
        Returns:
            dynamic_adj: 动态邻接矩阵 [B, N, N]
            attention_weights: 注意力权重 [B, N, N]
        """
        B, T, N, F = x.shape

        # 1. 节点静态嵌入
        node_ids = torch.arange(N, device=x.device)
        static_emb = self.node_embedding(node_ids)  # [N, D]
        static_emb = static_emb.unsqueeze(0).expand(B, N, -1)  # [B, N, D]

        # 2. 时序动态嵌入
        # 重塑为 [B*N, T, F]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, F)

        # GRU时序编码
        temporal_out, _ = self.temporal_encoder(x_reshaped)  # [B*N, T, temp_emb_dim]

        # 取最后seq_len个时间步的加权平均
        if T > seq_len:
            temporal_features = temporal_out[:, -seq_len:, :]  # [B*N, seq_len, D]
        else:
            temporal_features = temporal_out

        # 自适应时序注意力
        temporal_attention = torch.softmax(
            torch.matmul(temporal_features, temporal_features.transpose(1, 2)),
            dim=-1
        )  # [B*N, seq_len, seq_len]

        weighted_temporal = torch.matmul(temporal_attention, temporal_features)  # [B*N, seq_len, D]
        temporal_emb = weighted_temporal.mean(dim=1)  # [B*N, D]
        temporal_emb = temporal_emb.reshape(B, N, -1)  # [B, N, D]

        # 3. 动态边预测
        dynamic_adj = self._compute_dynamic_adjacency(static_emb, temporal_emb, static_adj)

        return dynamic_adj, temporal_attention.reshape(B, N, -1)

    def _compute_dynamic_adjacency(self, static_emb: torch.Tensor,
                                   temporal_emb: torch.Tensor,
                                   static_adj: torch.Tensor) -> torch.Tensor:
        """计算动态邻接矩阵"""
        B, N, D = static_emb.shape

        # 扩展节点特征用于成对计算
        static_emb_i = static_emb.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        static_emb_j = static_emb.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]

        temporal_emb_i = temporal_emb.unsqueeze(2).expand(B, N, N, -1)
        temporal_emb_j = temporal_emb.unsqueeze(1).expand(B, N, N, -1)

        # 拼接特征对
        pair_features = torch.cat([
            static_emb_i, static_emb_j,
            temporal_emb_i, temporal_emb_j
        ], dim=-1)  # [B, N, N, 4D]

        # 预测边权重
        edge_weights = self.edge_predictor(pair_features).squeeze(-1)  # [B, N, N]

        # 结合静态先验
        if static_adj.dim() == 2:
            static_adj = static_adj.unsqueeze(0).expand(B, N, N)

        # 应用静态图稀疏性约束
        dynamic_adj = edge_weights * static_adj

        # 对称化处理
        dynamic_adj = (dynamic_adj + dynamic_adj.transpose(1, 2)) / 2

        # 行归一化
        row_sum = dynamic_adj.sum(dim=-1, keepdim=True) + 1e-8
        dynamic_adj = dynamic_adj / row_sum

        return dynamic_adj


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
    """动态空间特征融合模块 - 修复版"""

    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int = 128,
                 num_branches: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_branches = num_branches
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 分支特定的适配器
        self.branch_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_branches)
        ])

        # 动态融合门控 - 修复维度
        # 输入维度: hidden_dim * num_branches + hidden_dim (节点嵌入)
        gate_input_dim = hidden_dim * num_branches + hidden_dim
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_branches),
            nn.Softmax(dim=-1)
        )

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, branch_outputs: list, node_embeddings: torch.Tensor):
        """
        Args:
            branch_outputs: 各分支输出列表，每个 [B, N, input_dim]
            node_embeddings: 节点嵌入 [B, N, hidden_dim] 或 [N, hidden_dim]
        Returns:
            融合后的特征 [B, N, input_dim]
        """
        B, N, _ = branch_outputs[0].shape

        # 分支特征适配
        adapted_features = []
        for i, (adapter, branch_out) in enumerate(zip(self.branch_adapters, branch_outputs)):
            adapted = adapter(branch_out)  # [B, N, hidden_dim]
            adapted_features.append(adapted)

        # 拼接所有分支特征
        concat_features = torch.cat(adapted_features, dim=-1)  # [B, N, hidden_dim * num_branches]

        # 添加节点位置信息
        if node_embeddings.dim() == 2:
            node_emb = node_embeddings.unsqueeze(0).expand(B, N, -1)
        else:
            node_emb = node_embeddings

        # 融合门控计算 - 确保维度匹配
        gate_input = torch.cat([concat_features, node_emb], dim=-1)  # [B, N, hidden_dim*(num_branches+1)]

        # 检查维度
        expected_dim = self.hidden_dim * (self.num_branches + 1)
        if gate_input.size(-1) != expected_dim:
            print(f"警告: gate_input维度应为{expected_dim}，实际为{gate_input.size(-1)}，进行适配")
            # 使用线性投影适配维度
            dim_adapter = nn.Linear(gate_input.size(-1), expected_dim).to(gate_input.device)
            gate_input = dim_adapter(gate_input)

        fusion_weights = self.fusion_gate(gate_input)  # [B, N, num_branches]

        # 加权融合
        fused_features = sum(
            weight.unsqueeze(-1) * feat
            for weight, feat in zip(fusion_weights.unbind(-1), adapted_features)
        )  # [B, N, hidden_dim]

        # 输出投影和残差连接
        output = self.output_projection(fused_features)
        output = self.norm(output + adapted_features[0])  # 残差连接到第一个分支

        return output, fusion_weights


class EnhancedDynamicATGCN_SpatialModule(nn.Module):
    """增强的动态空间特征提取模块"""

    def __init__(self, num_nodes: int, node_features: int,
                 gcn_hidden: int =128, out_feats: int = 128,
                 gcn_layers: int = 3, dropout: float = 0.1,
                 dynamic_hidden: int = 128, temp_emb_dim: int = 64):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_features = node_features

        # 动态图学习器
        self.dynamic_graph_learner = DynamicGraphLearner(
            num_nodes=num_nodes,
            node_features=node_features,
            hidden_dim=dynamic_hidden,
            temp_emb_dim=temp_emb_dim,
            dropout=dropout
        )

        # 多分支图卷积
        # 1. 静态地理图卷积
        self.static_geo_gat = LightweightGCNModule(
            in_feats=node_features, 
            hidden_feats=gcn_hidden, 
            out_feats=out_feats, 
            n_layers=gcn_layers, 
            dropout=dropout
        )


        # 2. 静态历史图卷积
        self.static_hist_gat = GATNetwork(
            node_features=node_features,
            hidden_features=gcn_hidden,
            out_features=out_feats,
            n_layers=gcn_layers,
            n_heads=8,
            dropout=dropout
        )

        # 3. 动态图卷积
        self.dynamic_gat = GATNetwork(
            node_features=node_features,
            hidden_features=gcn_hidden,
            out_features=out_feats,
            n_layers=gcn_layers,
            n_heads=8,
            dropout=dropout
        )

        # 4. 多尺度时序图卷积
        self.temporal_gcn = MultiScaleTemporalGCN(
            in_feats=node_features,
            out_feats=out_feats,
            temporal_scales=[3, 6, 12],
            dropout=dropout
        )

        # 动态特征融合
        self.fusion_module = DynamicSpatialFusion(
            num_nodes=num_nodes,
            input_dim=out_feats,
            hidden_dim=out_feats,
            num_branches=4,  # 4个分支
            dropout=dropout
        )

        # 输出适配器
        self.output_adapter = nn.Sequential(
            nn.Linear(out_feats, out_feats * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feats * 2, out_feats)
        )

        self.final_norm = nn.LayerNorm(out_feats)

    def forward(self, x: torch.Tensor, adj_geo: torch.Tensor,
                adj_hist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, seq_len, N, feat]
            adj_geo: 地理邻接矩阵 [N, N] 或 [B, N, N]
            adj_hist: 历史邻接矩阵 [N, N] 或 [B, N, N]
        Returns:
            空间特征 [B, seq_len, N, out_feats]
        """
        B, seq_len, N, feat = x.shape

        # 重塑输入用于处理
        x_reshaped = x.reshape(B * seq_len, N, feat)  # [B*seq_len, N, feat]

        # 1. 学习动态图结构
        x_for_dynamic = x.reshape(B, seq_len, N, feat)
        dynamic_adj, attention_weights = self.dynamic_graph_learner(
            x_for_dynamic, adj_geo, seq_len=min(seq_len, 12)
        )  # dynamic_adj: [B, N, N]

        # 扩展动态图到序列长度
        dynamic_adj_expanded = dynamic_adj.unsqueeze(1).expand(B, seq_len, N, N)
        dynamic_adj_expanded = dynamic_adj_expanded.reshape(B * seq_len, N, N)

        # 2. 多分支空间特征提取
        # 分支1: 静态地理图
        geo_features = self.static_geo_gat(x_reshaped, adj_geo)  # [B*seq_len, N, out_feats]

        # 分支2: 静态历史图
        hist_features = self.static_hist_gat(x_reshaped, adj_hist)  # [B*seq_len, N, out_feats]
        # hist_features = self.static_hist_gat(x_reshaped, dynamic_adj_expanded)

        # 分支3: 动态图
        dynamic_features = self.dynamic_gat(x_reshaped, dynamic_adj_expanded)  # [B*seq_len, N, out_feats]

        # 分支4: 多尺度时序图
        x_temporal = x.permute(0, 2, 1, 3)  # [B, N, seq_len, feat]
        temporal_features = self.temporal_gcn(x_temporal, adj_geo)  # [B, N, out_feats]
        # temporal_features = self.temporal_gcn(x_temporal, dynamic_adj_expanded)
        temporal_features = temporal_features.unsqueeze(1).expand(B, seq_len, N, -1)
        temporal_features = temporal_features.reshape(B * seq_len, N, -1)  # [B*seq_len, N, out_feats]

        # 3. 动态特征融合
        branch_outputs = [geo_features, hist_features, dynamic_features, temporal_features]

        # 获取节点嵌入用于融合
        node_embeddings = self.dynamic_graph_learner.node_embedding.weight  # [N, D]

        # 分批次处理融合（避免内存溢出）
        batch_size = 32  # 可根据GPU内存调整
        fused_features_list = []
        fusion_weights_list = []

        for i in range(0, B * seq_len, batch_size):
            end_idx = min(i + batch_size, B * seq_len)

            batch_branches = [branch[i:end_idx] for branch in branch_outputs]
            batch_node_emb = node_embeddings.unsqueeze(0).expand(end_idx - i, N, -1)

            fused_batch, weights_batch = self.fusion_module(
                batch_branches, batch_node_emb
            )

            fused_features_list.append(fused_batch)
            fusion_weights_list.append(weights_batch)

        fused_features = torch.cat(fused_features_list, dim=0)  # [B*seq_len, N, out_feats]

        # 4. 输出处理
        output = self.output_adapter(fused_features)
        output = self.final_norm(output)

        # 重塑回原始维度
        output = output.reshape(B, seq_len, N, -1)  # [B, seq_len, N, out_feats]

        return output


# 集成到主模型中的修改
class DynamicLightweightATGCN_InformerWithBiLSTM(nn.Module):
    """集成动态空间特征提取的完整模型"""
    
    def __init__(self, num_nodes, node_features, enc_in, dec_in, c_out=1,
                 seq_len=24, label_len=12, pred_len=1, d_model=128,
                 n_heads=4, e_layers=1, d_layers=1, d_ff=256,
                 dropout=0.1, lstm_hidden=128, lstm_layers=1,
                 gcn_hidden=64,  # 空间 GCN 隐层
                 gcn_layers=2,  # 空间 GCN 层数
                 dynamic_hidden=64,
                 temp_emb_dim=32, spatial_out=64):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # 时空嵌入
        self.st_embed = SpatioTemporalEmbed(num_nodes, spatial_out)
        
        # 趋势季节分解模块 - 新增
        self.ts_decomposition = TrendSeasonalDecomposition(num_nodes)
        
        # 趋势和季节的后续处理（可选，保持原有GRU处理）
        self.trend_informer_enc = InformerEncoderWithBiLSTM(
            d_model=spatial_out,  # 先保持 spatial_out 维，后面再投影到 d_model
            n_layers=e_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers
        )
        self.seasonal_informer_enc = InformerEncoderWithBiLSTM(
            d_model=spatial_out,
            n_layers=e_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers
        )
        
        # 后续处理的前馈网络（可选）
        self.ff_t = nn.Sequential(
            nn.Linear(spatial_out, spatial_out),
            nn.ReLU(),
            nn.Linear(spatial_out, spatial_out)
        )
        self.ff_s = nn.Sequential(
            nn.Linear(spatial_out, spatial_out),
            nn.ReLU(),
            nn.Linear(spatial_out, spatial_out)
        )
        
        # 动态空间特征提取模块
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
        
        # 2. 维度适配
        self.feature_adapter = nn.Sequential(
            nn.Linear(spatial_out, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 时序模块（保持原有结构）
        self.decoder = LSTMDecoder(
            d_model=d_model, pred_len=pred_len, n_layers=d_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, lstm_hidden=lstm_hidden, lstm_layers=max(1, lstm_layers // 2)
        )
        
        # 输入投影（用于解码器输入）
        self.input_proj = nn.Linear(node_features, d_model)
        self.trend_proj = nn.Linear(spatial_out, d_model)
        self.seasonal_proj = nn.Linear(spatial_out, d_model)
        
        print(f"动态模型参数量: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, adj_geo, adj_hist):
        # x_enc: [B, seq_len, N, feat]
        B, seq_len, N, feat = x_enc.shape
        
        # 1. 动态空间特征提取
        spatial_feat = self.spatial_module(x_enc, adj_geo, adj_hist)
        
        # 2. 时空嵌入
        M = self.st_embed(x_mark_enc)  # [B,T,N,D]
        
        # 3. 趋势季节分解（使用与model.py一致的方式）
        # 使用可学习的融合权重分解
        ts_feat, trend, seasonal = self.ts_decomposition(spatial_feat, M)
        
        # 4. 可选：对趋势和季节分别进行前馈处理（与model.py中的FeedForward一致）
        trend_ff = self.ff_t(trend)+trend
        seasonal_ff = self.ff_s(seasonal)+seasonal

        # 4. reshape → [B*N, T, spatial_out]  适配 Informer
        trend_rs = trend_ff.permute(0, 2, 1, 3).reshape(B * N, seq_len, -1)
        seasonal_rs = seasonal_ff.permute(0, 2, 1, 3).reshape(B * N, seq_len, -1)

        # 5. 分别用 BiLSTM-enhanced Informer 编码
        trend_enc = self.trend_informer_enc(trend_rs)  # [B*N, T, spatial_out]
        seasonal_enc = self.seasonal_informer_enc(seasonal_rs)  # [B*N, T, spatial_out]

        # 6. 融合 & 投影到 d_model
        ts_encoded = trend_enc + seasonal_enc  # [B*N, T, spatial_out]
        ts_encoded = self.trend_proj(ts_encoded)  # [B*N, T, d_model]

        # 7. 准备解码器输入
        dec_proj = self.input_proj(x_dec)  # [B, label_len+pred_len, N, d_model]
        dec_input = dec_proj.permute(0, 2, 1, 3).reshape(B * N, dec_proj.size(1), -1)

        # 8. 解码器预测
        output = self.decoder(ts_encoded, dec_input)  # [B*N, pred_len]

        # 9.  reshape 输出
        out = output.reshape(B, N, self.pred_len).permute(0, 2, 1)  # [B, pred_len, N]
        return out


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
        data_path="D:\Pycharm\PythonProject\data\data\PEMS04\pems04.npz",  # 替换为实际路径
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
    adj_geo = processor.load_adjacency_matrix("D:\Pycharm\PythonProject\data\data\PEMS04\distance4.csv")
    # 只使用训练数据计算历史相似度

    adj_hist = processor.compute_historical_similarity(original_data)  # ✅ 仅用训练数据
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