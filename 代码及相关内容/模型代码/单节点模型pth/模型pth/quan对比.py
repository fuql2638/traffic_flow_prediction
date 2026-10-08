import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
from typing import Tuple

warnings.filterwarnings('ignore')
import os
import math
from datetime import datetime, timedelta
from mstcn_bilstminfo import MSTCNInformerWithBiLSTM

# 导入ARIMA相关库
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# 设置中文字体支持
import matplotlib as mpl
import matplotlib.pyplot as plt

cmap = plt.cm.get_cmap('Set3')
try:
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False
    print("已设置中文字体支持")
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正常显示")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题[1,4](@ref)


# 设置随机种子确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

# GPU配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ... [保持原有的模型定义不变] ...
class BiLSTMModel(nn.Module):
    """双向LSTM模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.2, use_attention: bool = True):
        super(BiLSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

        # 注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim , hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # BiLSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim*2]

        if self.use_attention:
            # 注意力机制
            attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
            attention_weights = F.softmax(attention_weights, dim=1)

            # 加权求和
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        else:
            # 如果没有注意力，使用最后一个时间步的输出
            context_vector = lstm_out[:, -1, :]  # [batch_size, hidden_dim*2]

        # 全连接输出层
        output = self.fc(context_vector)  # [batch_size, output_dim]

        return output  # [batch_size]


# GRU模型定义
class GRUModel(nn.Module):
    """GRU模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.2, use_attention: bool = True):
        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # GRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # GRU前向传播
        gru_out, hidden = self.gru(x)  # gru_out: [batch_size, seq_len, hidden_dim]

        if self.use_attention:
            # 注意力机制
            attention_weights = self.attention(gru_out)  # [batch_size, seq_len, 1]
            attention_weights = F.softmax(attention_weights, dim=1)

            # 加权求和
            context_vector = torch.sum(attention_weights * gru_out, dim=1)  # [batch_size, hidden_dim]
        else:
            # 如果没有注意力，使用最后一个时间步的输出
            context_vector = gru_out[:, -1, :]  # [batch_size, hidden_dim]

        # 全连接输出层
        output = self.fc(context_vector)  # [batch_size, output_dim]

        return output

# Transformer模型定义
class TransformerModel(nn.Module):
    """标准Transformer模型"""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 output_dim: int = 1, dropout: float = 0.2):
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # 输入投影层，将输入维度映射到d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )

        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化输入投影层
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0.0)

        # 初始化输出层
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # 输入投影
        x = self.input_projection(x) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]

        # 添加位置编码
        x = self.pos_encoder(x)  # [batch_size, seq_len, d_model]

        # Transformer编码
        transformer_out = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]

        # 取最后一个时间步的输出
        last_output = transformer_out[:, -1, :]  # [batch_size, d_model]

        # 输出层
        output = self.output_layer(last_output)  # [batch_size, output_dim]

        return output


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 传统RNN模型定义
class RNNModel(nn.Module):
    """传统RNN模型（非LSTM/GRU）"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.2, use_attention: bool = True):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # 传统RNN层
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',  # 可以选择'tanh'或'relu'
            dropout=dropout if num_layers > 1 else 0
        )

        # 注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                # 使用正交初始化，这对RNN有好处
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]

        # RNN前向传播
        rnn_out, hidden = self.rnn(x)  # rnn_out: [batch_size, seq_len, hidden_dim]

        if self.use_attention:
            # 注意力机制
            attention_weights = self.attention(rnn_out)  # [batch_size, seq_len, 1]
            attention_weights = F.softmax(attention_weights, dim=1)

            # 加权求和
            context_vector = torch.sum(attention_weights * rnn_out, dim=1)  # [batch_size, hidden_dim]
        else:
            # 如果没有注意力，使用最后一个时间步的输出
            context_vector = rnn_out[:, -1, :]  # [batch_size, hidden_dim]

        # 全连接输出层
        output = self.fc(context_vector)  # [batch_size, output_dim]

        return output


# 修改ARIMA模型类中的fit方法
class ARIMAModel:
    """ARIMA时间序列模型"""

    def __init__(self, order=None, seasonal_order=None):
        """
        初始化ARIMA模型

        Parameters:
        -----------
        order : tuple, optional
            (p, d, q)非季节性参数
        seasonal_order : tuple, optional
            (P, D, Q, s)季节性参数
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fitted = None

    def fit(self, train_data, seasonal_period=24):
        """
        训练ARIMA模型

        Parameters:
        -----------
        train_data : array-like
            训练数据，可以是numpy数组或pandas Series
        seasonal_period : int
            季节性周期（小时数据通常为24）
        """
        print("训练ARIMA模型...")

        # 确保train_data是numpy数组
        if isinstance(train_data, pd.Series):
            train_data = train_data.values
        elif isinstance(train_data, pd.DataFrame):
            train_data = train_data.values.flatten()

        # 检查数据是否有效
        if len(train_data) < 10:
            raise ValueError(f"训练数据太短: {len(train_data)}，至少需要10个样本")

        # 如果未指定参数，使用自动ARIMA寻找最佳参数
        if self.order is None:
            print("使用自动ARIMA选择最佳参数...")
            try:
                # 使用pandas Series进行auto_arima
                train_series = pd.Series(train_data)
                auto_model = auto_arima(
                    train_series,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5,
                    d=None,  # 让auto_arima自动选择d
                    seasonal=True,
                    start_P=0, start_Q=0,
                    max_P=2, max_Q=2,
                    D=None,  # 让auto_arima自动选择D
                    m=seasonal_period,  # 季节性周期
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    n_jobs=-1  # 使用所有CPU核心
                )
                self.order = auto_model.order
                self.seasonal_order = auto_model.seasonal_order
                self.model = auto_model
                print(f"自动ARIMA选择参数: order={self.order}, seasonal_order={self.seasonal_order}")
                return self

            except Exception as e:
                print(f"自动ARIMA失败: {e}")
                print("尝试使用默认参数...")

        # 如果指定了参数或自动ARIMA失败，使用SARIMAX
        print("使用SARIMAX模型...")

        # 设置默认参数
        if self.order is None:
            self.order = (1, 1, 1)
        if self.seasonal_order is None:
            self.seasonal_order = (1, 1, 1, seasonal_period)

        try:
            # 使用SARIMAX模型
            self.model = sm.tsa.SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=True
            )
            self.model_fitted = self.model.fit(disp=False, maxiter=100)
            print(f"SARIMAX模型训练完成: order={self.order}, seasonal_order={self.seasonal_order}")

        except Exception as e:
            print(f"SARIMAX模型训练失败: {e}")
            # 尝试更简单的模型
            print("尝试更简单的ARIMA模型...")
            try:
                self.order = (1, 0, 1)  # 不使用差分
                self.seasonal_order = (0, 0, 0, 0)  # 无季节性
                self.model = sm.tsa.SARIMAX(
                    train_data,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.model_fitted = self.model.fit(disp=False)
                print(f"简单ARIMA模型训练完成: order={self.order}")
            except Exception as e2:
                print(f"简单ARIMA模型也失败: {e2}")
                # 最后的尝试：使用ARIMA而不是SARIMAX
                print("最后尝试：使用statsmodels的ARIMA模型...")
                try:
                    self.model = ARIMA(train_data, order=self.order)
                    self.model_fitted = self.model.fit()
                    print(f"ARIMA模型训练完成: order={self.order}")
                except Exception as e3:
                    print(f"所有ARIMA模型都失败: {e3}")
                    raise RuntimeError("无法训练ARIMA模型")

        return self

    def predict(self, steps, start_idx=None):
        """
        预测未来值

        Parameters:
        -----------
        steps : int
            预测步数
        start_idx : int, optional
            预测起始位置

        Returns:
        --------
        predictions : array
            预测值
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        try:
            # 如果是auto_arima模型
            if hasattr(self.model, 'predict'):
                if start_idx is not None:
                    # auto_arima的预测方式不同
                    return self.model.predict(n_periods=steps)
                else:
                    return self.model.predict(n_periods=steps)

            # 如果是SARIMAX模型
            elif self.model_fitted is not None:
                if start_idx is not None:
                    forecast = self.model_fitted.get_prediction(start=start_idx, end=start_idx + steps - 1)
                else:
                    forecast = self.model_fitted.get_forecast(steps=steps)

                predictions = forecast.predicted_mean
                return predictions.values if hasattr(predictions, 'values') else predictions

            # 如果是statsmodels ARIMA模型
            else:
                if start_idx is not None:
                    # ARIMA模型的预测需要更多处理
                    return self.model_fitted.forecast(steps=steps)
                else:
                    return self.model_fitted.forecast(steps=steps)

        except Exception as e:
            print(f"ARIMA预测失败: {e}")
            # 返回简单预测作为后备
            print("使用简单预测作为后备...")
            if hasattr(self, 'last_value'):
                return np.full(steps, self.last_value)
            else:
                return np.full(steps, np.mean(self.model_fitted.fittedvalues) if hasattr(self.model_fitted,
                                                                                         'fittedvalues') else 0)

    def evaluate(self, test_data, start_idx=None):
        """
        评估模型性能

        Parameters:
        -----------
        test_data : array-like
            测试数据
        start_idx : int, optional
            预测起始位置

        Returns:
        --------
        metrics : dict
            评估指标
        predictions : array
            预测值
        """
        steps = len(test_data)

        try:
            predictions = self.predict(steps, start_idx)
        except Exception as e:
            print(f"ARIMA评估失败: {e}")
            # 返回零预测
            predictions = np.zeros(steps)

        # 确保预测长度与测试数据一致
        if len(predictions) > len(test_data):
            predictions = predictions[:len(test_data)]
        elif len(predictions) < len(test_data):
            # 填充缺失值
            padding = np.full(len(test_data) - len(predictions), predictions[-1] if len(predictions) > 0 else 0)
            predictions = np.concatenate([predictions, padding])

        # 确保测试数据是numpy数组
        if isinstance(test_data, pd.Series):
            test_data_np = test_data.values
        elif isinstance(test_data, pd.DataFrame):
            test_data_np = test_data.values.flatten()
        else:
            test_data_np = test_data

        # 检查数据是否有效
        if len(test_data_np) == 0 or len(predictions) == 0:
            print("测试数据或预测数据为空")
            return {
                'MSE': float('inf'),
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'R2': -float('inf'),
                'MAPE': float('inf')
            }, predictions

        # 计算指标
        mse = mean_squared_error(test_data_np, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data_np, predictions)
        r2 = r2_score(test_data_np, predictions)

        # 避免除零错误
        test_data_safe = test_data_np.copy()
        zero_mask = test_data_safe == 0
        if np.any(zero_mask):
            test_data_safe[zero_mask] = 1e-8

        mape = np.mean(np.abs((test_data_safe - predictions) / test_data_safe)) * 100

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

        return metrics, predictions


# ... [保持原有的load_data函数和数据集类不变] ...
def load_data(file_path: str, seq_len: int = 24, label_len: int = 12, pred_len: int = 1,
              test_months: int = 6, val_ratio: float = 0.3) -> Tuple:
    """加载并预处理数据，返回训练集、验证集、测试集，确保无数据泄露"""

    print("开始加载数据...")

    # 1. 基础数据加载和基本处理
    df = pd.read_csv(file_path, parse_dates=['date_time'])

    # 检查时间序列顺序
    assert df['date_time'].is_monotonic_increasing, "时间序列未按时间排序"

    # 数值特征均值聚合，类别特征取众数
    num_core_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
    cat_features = ['holiday', 'weather_main']

    # 填充类别特征
    df[cat_features] = df[cat_features].fillna('Unknown')

    # 定义聚合策略
    agg_strategy = {col: 'mean' for col in num_core_features}
    agg_strategy.update({
        col: lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        for col in cat_features
    })

    # 按时间聚合
    df = df.groupby('date_time').agg(agg_strategy).reset_index()

    # 填补缺失时间戳
    df = df.set_index('date_time').asfreq('h').reset_index()

    # 2. 按时间划分数据集（最后6个月为测试集，剩余数据按7:3划分训练集和验证集）
    latest_time = df['date_time'].max()
    test_start_time = latest_time - pd.Timedelta(days=30 * test_months)

    # 创建时间掩码
    test_mask = df['date_time'] >= test_start_time
    test_df_raw = df[test_mask].copy()
    train_val_df_raw = df[~test_mask].copy()

    # 从剩余数据中划分验证集（30%）
    val_start_idx = int(len(train_val_df_raw) * (1 - val_ratio))
    val_df_raw = train_val_df_raw.iloc[val_start_idx:].copy()
    train_df_raw = train_val_df_raw.iloc[:val_start_idx].copy()

    print(f"训练集时间范围: {train_df_raw['date_time'].min()} 到 {train_df_raw['date_time'].max()}")
    print(f"验证集时间范围: {val_df_raw['date_time'].min()} 到 {val_df_raw['date_time'].max()}")
    print(f"测试集时间范围: {test_df_raw['date_time'].min()} 到 {test_df_raw['date_time'].max()}")
    print(f"数据集大小 - 训练集: {len(train_df_raw)}, 验证集: {len(val_df_raw)}, 测试集: {len(test_df_raw)}")

    # 3. 安全的数据预处理函数（避免数据泄露）
    def safe_preprocess_dataset(data, is_training=True, train_stats=None):
        """安全的数据预处理，避免数据泄露"""
        data = data.copy()
        num_features_list = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']

        if is_training:
            # 训练集：计算统计量
            num_means = data[num_features_list].mean()
            cat_modes = data[cat_features].mode().iloc[0] if not data[cat_features].mode().empty else 'Unknown'
        else:
            # 验证集/测试集：使用训练集的统计量
            num_means = train_stats['num_means']
            cat_modes = train_stats['cat_modes']

        # 数值特征插值（使用安全的方法）
        for col in num_features_list:
            data[col] = data[col].interpolate(method='linear')
            # 使用前向填充处理剩余的NaN
            data[col] = data[col].ffill().bfill()

        # 类别特征处理
        for col in cat_features:
            data[col] = data[col].fillna(cat_modes[col] if isinstance(cat_modes, dict) else cat_modes)

        # 安全的异常值检测（只使用历史信息）
        def safe_correct_anomalies(data_col):
            """使用扩展窗口而不是滚动窗口，避免未来信息泄露"""
            data_clean = data_col.copy()

            for i in range(1, len(data_clean)):
                if i >= 6:  # 至少有6个历史点
                    # 只使用历史数据计算统计量
                    historical_data = data_clean[:i]
                    hist_mean = historical_data.mean()
                    hist_std = historical_data.std()

                    # 避免除零
                    if hist_std == 0:
                        hist_std = 1e-8

                    # 检测异常值
                    current_val = data_clean.iloc[i]
                    z_score = abs(current_val - hist_mean) / hist_std

                    if z_score > 3:  # 3σ原则
                        # 使用历史均值修正异常值
                        data_clean.iloc[i] = hist_mean
                else:
                    # 对于前几个点，使用简单的阈值检测
                    current_val = data_clean.iloc[i]
                    if abs(current_val - data_clean.iloc[:i + 1].mean()) > 3 * data_clean.iloc[:i + 1].std():
                        data_clean.iloc[i] = data_clean.iloc[:i].mean()

            return data_clean

        # 应用安全的异常值检测（不处理目标变量traffic_volume）
        for col in num_features_list[:-1]:  # 排除traffic_volume
            data[col] = safe_correct_anomalies(data[col])

        if is_training:
            # 返回训练集统计量
            stats = {
                'num_means': num_means,
                'cat_modes': cat_modes
            }
            return data, stats
        else:
            return data

    # 预处理训练集
    print("预处理训练集...")
    train_df_processed, train_stats = safe_preprocess_dataset(train_df_raw, is_training=True)

    # 预处理验证集和测试集（使用训练集统计量）
    print("预处理验证集...")
    val_df_processed = safe_preprocess_dataset(val_df_raw, is_training=False, train_stats=train_stats)

    print("预处理测试集...")
    test_df_processed = safe_preprocess_dataset(test_df_raw, is_training=False, train_stats=train_stats)

    # 4. 特征工程（在划分后的数据集上分别进行）
    def add_features(data):
        """添加时间特征"""
        data = data.copy()
        data['date_time'] = pd.to_datetime(data['date_time'])
        data['hour'] = data['date_time'].dt.hour
        data['weekday'] = data['date_time'].dt.weekday
        data['month'] = data['date_time'].dt.month
        data['day'] = data['date_time'].dt.day
        data['is_weekend'] = (data['weekday'] >= 5).astype(int)

        # 添加周期性特征
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
        data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

        return data

    train_df_features = add_features(train_df_processed)
    val_df_features = add_features(val_df_processed)
    test_df_features = add_features(test_df_processed)

    # 5. 生成滞后特征（分别在训练集、验证集和测试集上）
    def add_lag_features(data, seq_len, is_training=True, train_lag_values=None):
        """添加滞后特征，避免数据泄露"""
        data = data.copy()

        if is_training:
            # 训练集：正常生成滞后特征
            for lag in range(1, seq_len + 1):
                data[f'traffic_lag_{lag}'] = data['traffic_volume'].shift(lag)

            # 保存最后几个值供验证集和测试集使用
            lag_values = {f'lag_{lag}': data['traffic_volume'].iloc[-lag:].values
                          for lag in range(1, seq_len + 1)}

            # 删除初始NaN行
            data = data.iloc[seq_len:].reset_index(drop=True)
            return data, lag_values
        else:
            # 验证集/测试集：使用训练集的最后几个值填充初始滞后
            for lag in range(1, seq_len + 1):
                lag_col = f'traffic_lag_{lag}'

                if lag <= len(train_lag_values[f'lag_{lag}']):
                    # 用训练集的最后几个值填充开头的滞后
                    test_lags = np.full(len(data), np.nan)

                    # 前lag个位置用训练集的值
                    train_vals = train_lag_values[f'lag_{lag}']
                    test_lags[:lag] = train_vals[-lag:] if lag <= len(train_vals) else train_vals

                    # 剩余位置用自身的历史
                    for i in range(lag, len(data)):
                        test_lags[i] = data['traffic_volume'].iloc[i - lag]

                    data[lag_col] = test_lags
                else:
                    data[lag_col] = data['traffic_volume'].shift(lag)

            # 删除NaN行
            data = data.iloc[seq_len:].reset_index(drop=True)
            return data

    # 训练集添加滞后特征
    print("生成训练集滞后特征...")
    train_df_final, train_lag_values = add_lag_features(train_df_features, seq_len, is_training=True)

    # 验证集添加滞后特征（使用训练集的滞后值）
    print("生成验证集滞后特征...")
    val_df_final = add_lag_features(val_df_features, seq_len, is_training=False,
                                    train_lag_values=train_lag_values)

    # 测试集添加滞后特征（使用训练集的滞后值）
    print("生成测试集滞后特征...")
    test_df_final = add_lag_features(test_df_features, seq_len, is_training=False,
                                     train_lag_values=train_lag_values)

    # 6. 特征编码和归一化（使用训练集统计量）
    print("特征编码和归一化...")

    # 定义特征组
    num_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'weekday', 'month', 'day', 'is_weekend',
                    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']
    lag_features = [f'traffic_lag_{i}' for i in range(1, seq_len + 1)]
    all_num_features = num_features + lag_features
    all_cat_features = ['holiday', 'weather_main']
    time_features = ['hour', 'weekday', 'month', 'day', 'is_weekend',
                     'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']

    # 只在训练集上拟合编码器和归一化器
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded_train = encoder.fit_transform(train_df_final[all_cat_features])

    scaler_num = MinMaxScaler()
    scaled_num_train = scaler_num.fit_transform(train_df_final[all_num_features])

    scaler_target = MinMaxScaler()
    scaled_target_train = scaler_target.fit_transform(train_df_final[['traffic_volume']])

    # 对验证集和测试集应用训练集的转换
    cat_encoded_val = encoder.transform(val_df_final[all_cat_features])
    scaled_num_val = scaler_num.transform(val_df_final[all_num_features])
    scaled_target_val = scaler_target.transform(val_df_final[['traffic_volume']])

    cat_encoded_test = encoder.transform(test_df_final[all_cat_features])
    scaled_num_test = scaler_num.transform(test_df_final[all_num_features])
    scaled_target_test = scaler_target.transform(test_df_final[['traffic_volume']])

    # 合并特征
    dataset_train = np.hstack([scaled_num_train, cat_encoded_train])
    dataset_val = np.hstack([scaled_num_val, cat_encoded_val])
    dataset_test = np.hstack([scaled_num_test, cat_encoded_test])

    # 时间特征（不需要归一化）
    time_train = train_df_final[time_features].values
    time_val = val_df_final[time_features].values
    time_test = test_df_final[time_features].values

    # 数据完整性检查
    assert not np.isnan(dataset_train).any(), "训练集存在NaN值"
    assert not np.isnan(dataset_val).any(), "验证集存在NaN值"
    assert not np.isnan(dataset_test).any(), "测试集存在NaN值"

    # 7. 构建Informer时序样本
    '''def create_informer_sequences(data, time_data, targets, seq_len, label_len, pred_len):
        X_enc, X_mark_enc, X_dec, X_mark_dec, Y = [], [], [], [], []

        for i in range(len(data) - seq_len - pred_len + 1):
            # 编码器输入
            enc_start = i
            enc_end = enc_start + seq_len
            x_enc = data[enc_start:enc_end]
            x_mark_enc = time_data[enc_start:enc_end]

            # 解码器输入
            dec_start = enc_end - label_len
            dec_end = enc_end + pred_len
            x_dec = data[dec_start:dec_end]
            #x_dec_history = data[dec_start:enc_end]  # 只包含历史
            #x_dec_future = np.zeros((pred_len, data.shape[1]))  # 用0填充未来特征
            #x_dec = np.concatenate([x_dec_history, x_dec_future], axis=0)
            x_mark_dec = time_data[dec_start:dec_end]


            # 目标值
            y = targets[enc_end:enc_end + pred_len].flatten()

            X_enc.append(x_enc)
            X_mark_enc.append(x_mark_enc)
            X_dec.append(x_dec)
            X_mark_dec.append(x_mark_dec)
            Y.append(y)

        return np.array(X_enc), np.array(X_mark_enc), np.array(X_dec), np.array(X_mark_dec), np.array(Y)

    def create_informer_sequences(data, time_data, targets, seq_len, label_len, pred_len):
        X_enc, X_mark_enc, X_dec, X_mark_dec, Y = [], [], [], [], []

        for i in range(len(data) - seq_len - pred_len + 1):
            # 编码器输入
            enc_start = i
            enc_end = enc_start + seq_len
            x_enc = data[enc_start:enc_end]
            x_mark_enc = time_data[enc_start:enc_end]

            # 解码器输入 - 只使用历史部分
            dec_start = enc_end - label_len
            dec_end = enc_end
            x_dec_history = data[dec_start:dec_end]

            # 未来部分用0填充，但时间特征只保留历史部分
            x_dec_future = np.zeros((pred_len, data.shape[1]))
            x_dec = np.concatenate([x_dec_history, x_dec_future], axis=0)

            # 时间特征：历史部分使用真实值，未来部分用0或特殊标记
            x_mark_dec_history = time_data[dec_start:dec_end]
            x_mark_dec_future = np.zeros((pred_len, time_data.shape[1]))
            x_mark_dec = np.concatenate([x_mark_dec_history, x_mark_dec_future], axis=0)

            # 目标值
            y = targets[enc_end:enc_end + pred_len].flatten()

            X_enc.append(x_enc)
            X_mark_enc.append(x_mark_enc)
            X_dec.append(x_dec)
            X_mark_dec.append(x_mark_dec)
            Y.append(y)

        return np.array(X_enc), np.array(X_mark_enc), np.array(X_dec), np.array(X_mark_dec), np.array(Y)'''

    def create_informer_sequences(data, time_data, targets, seq_len, label_len, pred_len):
        X_enc, X_mark_enc, X_dec, X_mark_dec, Y = [], [], [], [], []

        for i in range(len(data) - seq_len - pred_len + 1):
            # 编码器输入（使用历史信息）
            enc_start = i
            enc_end = enc_start + seq_len
            x_enc = data[enc_start:enc_end]
            x_mark_enc = time_data[enc_start:enc_end]

            # 解码器输入 - 严格使用历史信息
            # 只使用编码器最后label_len个时间步的信息
            dec_history_start = enc_end - label_len
            dec_history_end = enc_end
            x_dec_history = data[dec_history_start:dec_history_end]

            # 未来部分完全用0填充（模拟真实预测场景）
            x_dec_future = np.zeros((pred_len, data.shape[1]))
            x_dec = np.concatenate([x_dec_history, x_dec_future], axis=0)

            # 时间特征：历史部分使用真实时间，未来部分使用已知的未来时间
            x_mark_dec_history = time_data[dec_history_start:dec_history_end]
            # 未来时间特征是已知的，可以使用真实未来时间
            x_mark_dec_future = time_data[enc_end:enc_end + pred_len]
            x_mark_dec = np.concatenate([x_mark_dec_history, x_mark_dec_future], axis=0)

            # 目标值
            y = targets[enc_end:enc_end + pred_len].flatten()

            X_enc.append(x_enc)
            X_mark_enc.append(x_mark_enc)
            X_dec.append(x_dec)
            X_mark_dec.append(x_mark_dec)
            Y.append(y)

        return np.array(X_enc), np.array(X_mark_enc), np.array(X_dec), np.array(X_mark_dec), np.array(Y)

    # 创建训练集、验证集和测试集序列
    print("创建训练集序列样本...")
    X_enc_train, X_mark_enc_train, X_dec_train, X_mark_dec_train, y_train = create_informer_sequences(
        dataset_train, time_train, scaled_target_train, seq_len, label_len, pred_len
    )

    print("创建验证集序列样本...")
    X_enc_val, X_mark_enc_val, X_dec_val, X_mark_dec_val, y_val = create_informer_sequences(
        dataset_val, time_val, scaled_target_val, seq_len, label_len, pred_len
    )

    print("创建测试集序列样本...")
    X_enc_test, X_mark_enc_test, X_dec_test, X_mark_dec_test, y_test = create_informer_sequences(
        dataset_test, time_test, scaled_target_test, seq_len, label_len, pred_len
    )

    print(f"训练集形状: X_enc {X_enc_train.shape}, X_mark_enc {X_mark_enc_train.shape}, "
          f"X_dec {X_dec_train.shape}, X_mark_dec {X_mark_dec_train.shape}, y {y_train.shape}")
    print(f"验证集形状: X_enc {X_enc_val.shape}, X_mark_enc {X_mark_enc_val.shape}, "
          f"X_dec {X_dec_val.shape}, X_mark_dec {X_mark_dec_val.shape}, y {y_val.shape}")
    print(f"测试集形状: X_enc {X_enc_test.shape}, X_mark_enc {X_mark_enc_test.shape}, "
          f"X_dec {X_dec_test.shape}, X_mark_dec {X_mark_dec_test.shape}, y {y_test.shape}")

    return (X_enc_train, X_mark_enc_train, X_dec_train, X_mark_dec_train, y_train,
            X_enc_val, X_mark_enc_val, X_dec_val, X_mark_dec_val, y_val,
            X_enc_test, X_mark_enc_test, X_dec_test, X_mark_dec_test, y_test,
            scaler_target, train_df_final['traffic_volume'].values)
# 数据集类
class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y):
        self.x_enc = torch.FloatTensor(x_enc)
        self.x_mark_enc = torch.FloatTensor(x_mark_enc)
        self.x_dec = torch.FloatTensor(x_dec)
        self.x_mark_dec = torch.FloatTensor(x_mark_dec)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.x_enc)

    def __getitem__(self, idx):
        return (self.x_enc[idx], self.x_mark_enc[idx],
                self.x_dec[idx], self.x_mark_dec[idx], self.y[idx])

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in data_loader:
            # 统一把 batch 解包，不管用不用
            batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = batch
            batch_x_enc = batch_x_enc.to(device)
            batch_y   = batch_y.to(device)

            # ① Informer 系列模型需要 4 个输入
            if model.__class__.__name__ in {'MSTCNInformerWithBiLSTM'}:
                batch_x_mark_enc = batch_x_mark_enc.to(device)
                batch_x_dec      = batch_x_dec.to(device)
                batch_x_mark_dec = batch_x_mark_dec.to(device)
                outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            else:
                # ② 其它模型只要一个 x
                outputs = model(batch_x_enc)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            all_pred.append(outputs.cpu().numpy())
            all_true.append(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, np.concatenate(all_pred), np.concatenate(all_true)


# 修改主函数部分，加入ARIMA模型
def main():
    # 超参数设置
    SEQ_LENGTH = 24
    LABEL_LENGTH = 12
    PRED_LEN = 1
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.001
    PATIENCE = 15

    # 创建保存目录
    save_dir = r'D:\python-anaconda-learn1\交通流量\Alast-final\图表'
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print("开始加载数据...")
    (X_enc_train, X_mark_enc_train, X_dec_train, X_mark_dec_train, y_train,
     X_enc_val, X_mark_enc_val, X_dec_val, X_mark_dec_val, y_val,
     X_enc_test, X_mark_enc_test, X_dec_test, X_mark_dec_test, y_test,
     scaler_target, train_traffic) = load_data("州际交通量数据集.csv",
                                               seq_len=SEQ_LENGTH,
                                               label_len=LABEL_LENGTH,
                                               pred_len=PRED_LEN,
                                               test_months=6,
                                               val_ratio=0.2)

    # 计算输入维度
    enc_in = X_enc_train.shape[2]
    dec_in = X_dec_train.shape[2]
    print(f"编码器输入维度: {enc_in}, 解码器输入维度: {dec_in}")

    # 创建数据集
    train_dataset = TrafficDataset(X_enc_train, X_mark_enc_train, X_dec_train, X_mark_dec_train, y_train)
    val_dataset = TrafficDataset(X_enc_val, X_mark_enc_val, X_dec_val, X_mark_dec_val, y_val)
    test_dataset = TrafficDataset(X_enc_test, X_mark_enc_test, X_dec_test, X_mark_dec_test, y_test)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                              pin_memory=True)

    # 反归一化函数
    def inverse_transform(scaler, pred):
        dummy = np.zeros((len(pred), 1))
        dummy[:, 0] = pred.ravel()
        return scaler.inverse_transform(dummy).ravel()

    # 为ARIMA准备数据
    print("\n准备ARIMA模型数据...")

    # 合并训练集和验证集作为ARIMA的训练数据
    # 获取训练集和验证集的反归一化数据
    y_train_original = inverse_transform(scaler_target, y_train)
    y_val_original = inverse_transform(scaler_target, y_val)

    # 对于ARIMA，我们需要连续的时间序列数据
    # 由于我们的数据集是序列化的，需要重建原始时间序列
    # 这里我们使用训练集的最后一个时间步作为ARIMA的预测起点

    # 训练ARIMA模型
    print("开始训练ARIMA模型...")

    # 为ARIMA准备训练数据（合并训练集和验证集）
    # 注意：由于我们的数据已经序列化，我们需要重新加载原始连续数据
    # 这里简化处理，直接使用反归一化后的测试集真实值进行对比

    # 获取测试集真实值（反归一化）
    test_true_original = inverse_transform(scaler_target, y_test)

    # 由于ARIMA需要连续的时间序列，我们需要从原始数据中获取
    # 这里我们模拟一个简单的ARIMA预测流程

    # 加载原始数据用于ARIMA
    df = pd.read_csv("州际交通量数据集.csv", parse_dates=['date_time'])
    df = df.groupby('date_time')['traffic_volume'].mean().reset_index()
    df = df.set_index('date_time').asfreq('h').reset_index()

    # 填充缺失值
    df['traffic_volume'] = df['traffic_volume'].interpolate(method='linear').ffill().bfill()

    # 划分训练集和测试集（与神经网络相同的划分方式）
    test_months = 6
    latest_time = df['date_time'].max()
    test_start_time = latest_time - pd.Timedelta(days=30 * test_months)

    train_val_df = df[df['date_time'] < test_start_time]
    test_df = df[df['date_time'] >= test_start_time]

    # 进一步划分训练集和验证集
    val_ratio = 0.2
    val_start_idx = int(len(train_val_df) * (1 - val_ratio))
    train_df = train_val_df.iloc[:val_start_idx]
    val_df = train_val_df.iloc[val_start_idx:]

    print(f"ARIMA数据划分 - 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

    # 训练ARIMA模型（使用训练集）
    arima_train_data = train_df['traffic_volume']

    # 尝试不同的ARIMA参数
    arima_orders = [
        (1, 1, 1),  # 基本ARIMA
        (2, 1, 2),  # 更复杂的ARIMA
        (1, 1, 0),  # AR模型
        (0, 1, 1),  # MA模型
        (3, 1, 3),  # 更复杂的模型
    ]

    best_arima_model = None
    best_arima_val_loss = float('inf')
    best_arima_order = None

    # 使用验证集选择最佳ARIMA参数
    for order in arima_orders:
        try:
            print(f"\n尝试ARIMA{order}...")
            arima_model = ARIMAModel(order=order, seasonal_order=None)
            arima_model.fit(arima_train_data)

            # 在验证集上评估
            val_predictions_start = len(arima_train_data)
            val_metrics, val_pred = arima_model.evaluate(
                val_df['traffic_volume'],
                start_idx=val_predictions_start
            )

            print(f"ARIMA{order} 验证集RMSE: {val_metrics['RMSE']:.4f}")

            if val_metrics['RMSE'] < best_arima_val_loss:
                best_arima_val_loss = val_metrics['RMSE']
                best_arima_model = arima_model
                best_arima_order = order

        except Exception as e:
            print(f"ARIMA{order} 训练失败: {e}")
            continue

    if best_arima_model is None:
        print("所有ARIMA模型训练失败，使用默认ARIMA(1,1,1)")
        best_arima_model = ARIMAModel(order=(1, 1, 1), seasonal_order=None)
        best_arima_model.fit(arima_train_data)
        best_arima_order = (1, 1, 1)

    print(f"\n最佳ARIMA模型: ARIMA{best_arima_order}")

    # 在测试集上预测
    print("在测试集上运行ARIMA预测...")

    # 合并训练集和验证集作为最终训练数据
    final_train_data = np.concatenate([train_df['traffic_volume'].values,
                                       val_df['traffic_volume'].values])

    # 重新训练最佳模型
    best_arima_model.fit(final_train_data)

    # 预测测试集
    test_predictions_start = len(final_train_data)
    arima_test_metrics, arima_test_pred = best_arima_model.evaluate(
        test_df['traffic_volume'].values,
        start_idx=test_predictions_start
    )

    print(f"ARIMA测试集结果:")
    print(f"  MSE: {arima_test_metrics['MSE']:.4f}, RMSE: {arima_test_metrics['RMSE']:.4f}")
    print(f"  MAE: {arima_test_metrics['MAE']:.4f}, R2: {arima_test_metrics['R2']:.4f}")
    print(f"  MAPE: {arima_test_metrics['MAPE']:.4f}%")

    # 存储ARIMA结果
    arima_results = {
        'predictions': arima_test_pred,
        'true_values': test_df['traffic_volume'].values,
        'metrics': arima_test_metrics
    }

    # 继续原有的神经网络模型训练和评估
    print("\n" + "=" * 60)
    print("开始神经网络模型对比训练...")
    print("=" * 60)

    # 定义所有要训练的模型（保持不变）
    models_dict = {
        'MSTCN-BiLSTM-Informer': MSTCNInformerWithBiLSTM(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=1,
            seq_len=SEQ_LENGTH,
            label_len=LABEL_LENGTH,
            pred_len=1,
            d_model=256,
            n_heads=8,
            e_layers=2,
            d_layers=1,
            d_ff=512,
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.2,
            mstcn_hidden=128,
            mstcn_kernels=[6, 12, 24]
        ).to(device),
        'LSTM': BiLSTMModel(input_dim=enc_in, hidden_dim=128, num_layers=2,
                            output_dim=1, dropout=0.2, use_attention=False).to(device),
        'GRU': GRUModel(input_dim=enc_in, hidden_dim=128, num_layers=2,
                        output_dim=1, dropout=0.2, use_attention=False).to(device),
        'Transformer': TransformerModel(input_dim=enc_in, d_model=128, nhead=4,
                                        num_encoder_layers=1, dim_feedforward=1024,
                                        output_dim=1, dropout=0.2).to(device),
        'RNN': RNNModel(input_dim=enc_in, hidden_dim=128, num_layers=2,
                        output_dim=1, dropout=0.2, use_attention=False).to(device)
    }

    # 权重目录
    WEIGHT_DIR = r'D:\python-anaconda-learn1\交通流量\十二月\模型pth'
    WEIGHT_FILES = {
        'MSTCN-BiLSTM-Informer': os.path.join(WEIGHT_DIR, 'MSTCN-BiLSTM-Informer_best_model.pth'),
        'LSTM': os.path.join(WEIGHT_DIR, 'lstm_model.pth'),
        'GRU': os.path.join(WEIGHT_DIR, 'gru_model.pth'),
        'RNN': os.path.join(WEIGHT_DIR, 'RNN_model.pth'),
        'Transformer': os.path.join(WEIGHT_DIR, 'Transformer_model.pth'),
    }

    # 存储所有模型的训练结果
    all_models_results = {}

    # 添加ARIMA结果
    all_models_results['ARIMA'] = arima_results

    # 训练或加载函数（保持不变）
    def train_or_load(model, model_name, train_loader, val_loader, criterion, device,
                      epochs=EPOCHS, lr=LR, patience=PATIENCE):
        weight_path = WEIGHT_FILES[model_name]

        if os.path.isfile(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f'[{model_name}] 权重已存在，直接加载 ← {weight_path}')
            return

        # 训练代码（保持不变）
        print(f'[{model_name}] 未找到权重，开始训练 …')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        best_loss, patience_counter = float('inf'), 0

        for epoch in range(epochs):
            # 训练
            model.train()
            epoch_loss = 0
            for *x, y in train_loader:
                x = [t.to(device) for t in x]
                y = y.to(device)
                optimizer.zero_grad()
                if model_name == 'MSTCN-BiLSTM-Informer':
                    out = model(*x)
                else:
                    out = model(x[0])
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_train = epoch_loss / len(train_loader)

            # 验证
            val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), weight_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'[{model_name}] 早停 @ epoch {epoch + 1}')
                    break

            if (epoch + 1) % 10 == 0:
                print(f'[{model_name}] Epoch {epoch + 1}  train {avg_train:.5f}  val {val_loss:.5f}')

        model.load_state_dict(torch.load(weight_path, map_location=device))

    # 训练每个神经网络模型
    criterion = nn.MSELoss()
    for model_name, model_instance in models_dict.items():
        print('\n' + '=' * 60)
        print(f'开始处理模型：{model_name}')
        print('=' * 60)

        # 训练或加载
        train_or_load(model_instance, model_name,
                      train_loader, val_loader, criterion, device)

        # 测试集评估
        test_loss, pred_scaled, true_scaled = evaluate_model(
            model_instance, test_loader, criterion, device)

        # 反归一化
        test_pred_original = inverse_transform(scaler_target, pred_scaled)
        test_true_original = inverse_transform(scaler_target, true_scaled)

        # 计算指标
        mse = mean_squared_error(test_true_original, test_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_true_original, test_pred_original)
        r2 = r2_score(test_true_original, test_pred_original)
        mape = mean_absolute_percentage_error(test_true_original, test_pred_original)

        # 存储结果
        all_models_results[model_name] = {
            'predictions': test_pred_original,
            'true_values': test_true_original,
            'metrics': {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'Test Loss': test_loss
            }
        }

        print(f"{model_name} 测试结果:")
        print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}, MAPE: {mape:.4f}%")

    # 打印所有模型对比结果
    print("\n" + "=" * 60)
    print("所有模型性能对比（包括ARIMA）")
    print("=" * 60)

    # 创建对比表格
    metrics_table = []
    for model_name, results in all_models_results.items():
        metrics = results['metrics']
        if model_name == 'ARIMA':
            metrics_table.append([
                model_name,
                f"{metrics['MSE']:.4f}",
                f"{metrics['RMSE']:.4f}",
                f"{metrics['MAE']:.4f}",
                f"{metrics['R2']:.4f}",
                f"{metrics['MAPE']:.4f}%"
            ])
        else:
            metrics_table.append([
                model_name,
                f"{metrics['MSE']:.4f}",
                f"{metrics['RMSE']:.4f}",
                f"{metrics['MAE']:.4f}",
                f"{metrics['R2']:.4f}",
                f"{metrics['MAPE']:.4f}%"
            ])

    # 打印表格
    from tabulate import tabulate
    headers = ["模型", "MSE", "RMSE", "MAE", "R²", "MAPE"]
    print(tabulate(metrics_table, headers=headers, tablefmt="grid"))

    # 可视化所有模型的预测对比
    plt.figure(figsize=(16, 8))

    # 获取前N个样本（确保所有模型都有足够的数据）
    num_samples = min(48, len(all_models_results['MSTCN-BiLSTM-Informer']['true_values']))
    x_axis = range(num_samples)

    # 绘制真实值（使用第一个模型的真实值）
    true_values_key = list(all_models_results.keys())[0]
    plt.plot(x_axis, all_models_results[true_values_key]['true_values'][:num_samples],
             label='真实值', linewidth=3, color='black', linestyle='-', marker='o', markersize=3)

    # 定义颜色和线型
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    linestyles = ['-', '--', '--', '--', '--', '-.', ':']
    markers = ['o', 's', '^', 'd', 'v', '*', 'x']

    # 绘制各模型预测值
    for idx, (model_name, results) in enumerate(all_models_results.items()):
        # 确保预测数据长度足够
        pred_data = results['predictions']
        if len(pred_data) >= num_samples:
            plt.plot(x_axis, pred_data[:num_samples],
                     label=f'{model_name}预测',
                     linewidth=2,
                     color=colors[idx % len(colors)],
                     linestyle=linestyles[idx % len(linestyles)],
                     marker=markers[idx % len(markers)],
                     markersize=2,
                     alpha=0.8)

    plt.title('不同模型预测效果对比（包括ARIMA）', fontsize=16, fontweight='bold')
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('交通流量', fontsize=12)
    plt.legend(fontsize=10, loc='upper right', ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存对比图
    plt.savefig(os.path.join(save_dir, 'all_models_comparison_with_ARIMA.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制模型性能对比柱状图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
    metric_titles = ['MSE (越低越好)', 'RMSE (越低越好)', 'MAE (越低越好)',
                     'R² (越高越好)', 'MAPE% (越低越好)']

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        ax = axes[idx]

        model_names = list(all_models_results.keys())
        values = []

        for name in model_names:
            if metric == 'MAPE':
                # 处理MAPE的百分号
                map_val = all_models_results[name]['metrics'][metric]
                if isinstance(map_val, str):
                    val = float(map_val.replace('%', ''))
                else:
                    val = map_val
                values.append(val)
            else:
                values.append(all_models_results[name]['metrics'][metric])

        bars = ax.bar(model_names, values, color=plt.cm.Set3(range(len(model_names))))
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('值', fontsize=10)
        ax.tick_params(axis='x', rotation=45)

        # 在柱子上添加数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(values),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # 第六个子图：模型训练时间或复杂度对比
    ax = axes[5]
    # 这里可以添加模型复杂度或训练时间的对比
    # 暂时留空或添加其他信息
    ax.text(0.5, 0.5, '模型复杂度对比\n（可自行添加）',
            ha='center', va='center', fontsize=12)
    ax.set_title('模型复杂度', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.suptitle('各模型性能指标对比（包括ARIMA）', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'models_performance_metrics_with_ARIMA.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 保存所有模型的预测结果到CSV
    for model_name, results in all_models_results.items():
        df_results = pd.DataFrame({
            'True_Value': results['true_values'],
            'Predicted_Value': results['predictions'],
            'Error': results['true_values'] - results['predictions']
        })
        df_results.to_csv(os.path.join(save_dir, f'{model_name}_predictions.csv'), index=False)
        print(f"{model_name}的预测结果已保存")

    # 保存汇总的性能指标
    summary_data = {}
    for model_name, results in all_models_results.items():
        summary_data[model_name] = results['metrics']

    summary_df = pd.DataFrame(summary_data).T
    summary_df.to_csv(os.path.join(save_dir, 'models_performance_summary_with_ARIMA.csv'))
    print("\n所有模型的性能指标汇总已保存（包括ARIMA）")

    # 打印排名
    print("\n" + "=" * 60)
    print("模型性能排名")
    print("=" * 60)

    # 按RMSE排序
    rmse_scores = {model: results['metrics']['RMSE'] for model, results in all_models_results.items()}
    sorted_models = sorted(rmse_scores.items(), key=lambda x: x[1])

    print("按RMSE排名（越低越好）:")
    for rank, (model, rmse) in enumerate(sorted_models, 1):
        print(f"{rank}. {model}: {rmse:.4f}")

    # 按R²排序
    r2_scores = {model: results['metrics']['R2'] for model, results in all_models_results.items()}
    sorted_models_r2 = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n按R²排名（越高越好）:")
    for rank, (model, r2) in enumerate(sorted_models_r2, 1):
        print(f"{rank}. {model}: {r2:.4f}")


if __name__ == "__main__":
    main()