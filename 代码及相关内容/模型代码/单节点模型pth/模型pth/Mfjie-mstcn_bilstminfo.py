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

# 设置中文字体支持
import matplotlib as mpl

try:
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False
    print("已设置中文字体支持")
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正常显示")


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


# 评估指标函数
def smape(y_true, y_pred):
    """对称平均绝对百分比误差"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def mase(y_true, y_pred, y_train, seasonality=24):
    """平均绝对缩放误差"""
    naive_errors = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    return np.mean(np.abs(y_true - y_pred)) / naive_errors


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


# === 1. 修改SpatioTemporalEmbed模块，适应11维时间特征 ===
class SpatioTemporalEmbed(nn.Module):
    """时空嵌入模块 - 生成用于分解的M矩阵，适应现有数据格式"""

    def __init__(self, time_feat_dim, d_model):
        super().__init__()
        self.time_feat_dim = time_feat_dim

        # 根据实际时间特征维度调整MLP
        # 如果时间特征已经包含hour和weekday信息，直接映射
        self.time_mlp = nn.Sequential(
            nn.Linear(time_feat_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 输出激活函数，保证输出在0~1之间
        self.output_activation = nn.Sigmoid()

    def forward(self, x_mark_enc):
        """
        x_mark_enc: [B, T, time_feat_dim] 时间特征，当前为11维
        返回 M ∈ [B, T, D] 0~1，用于分解
        """
        # 直接映射时间特征
        M = self.time_mlp(x_mark_enc)  # [B, T, d_model]
        M = self.output_activation(M)  # 保证0~1

        return M


# === 2. STDN风格的趋势季节分解模块 ===
class STDN_TrendSeasonalDecomposition(nn.Module):
    """STDN风格的分解：使用时空嵌入M作为"刀子"进行分解"""

    def __init__(self, d_model):
        super().__init__()
        # 不需要可学习的vector，因为M会作为输入

    def forward(self, X, M):
        """
        X: [B, T, D] 输入特征
        M: [B, T, D] 时空嵌入，0~1，用于分解
        返回: trend, seasonal
        """
        # STDN分解公式
        trend = torch.mul(X, M)  # 趋势分量 = X * M
        seasonal = X - trend  # 季节分量 = X - trend

        return trend, seasonal


# === 2. 修改数据预处理，确保时间特征格式正确 ===
def prepare_time_features_for_stdn(x_mark_enc):
    """
    将现有的11维时间特征转换为适合STDN的格式
    假设x_mark_enc包含以下特征（11维）：
    0: hour (0-23, 归一化到0-1)
    1: weekday (0-6, 归一化到0-1)
    2: month (0-11, 归一化到0-1)
    3: day (1-31, 归一化到0-1)
    4: is_weekend (0或1)
    5: hour_sin (正弦编码)
    6: hour_cos (余弦编码)
    7: weekday_sin (正弦编码)
    8: weekday_cos (余弦编码)
    9: month_sin (正弦编码)
    10: month_cos (余弦编码)

    我们需要将其转换为31维（24小时one-hot + 7天星期one-hot）
    但为了保持一致性，我们可以使用原始特征，不需要one-hot
    """
    # 直接返回原始特征，让模型自行学习
    # 或者我们可以提取hour和weekday信息进行one-hot编码
    return x_mark_enc


# === 3. 更精确的转换函数（如果确实需要31维one-hot） ===
def convert_to_stdn_onehot(x_mark_enc):
    """
    将11维时间特征转换为STDN风格的31维one-hot编码
    假设前2个维度是hour和weekday的归一化值
    """
    import torch.nn.functional as F

    batch_size, seq_len, _ = x_mark_enc.shape

    # 提取hour和weekday（假设是前2个特征）
    hour_norm = x_mark_enc[..., 0]  # 归一化的hour
    weekday_norm = x_mark_enc[..., 1]  # 归一化的weekday

    # 反归一化：还原到原始范围
    # 假设hour是归一化到0-1的原始范围0-23
    hour = torch.clamp((hour_norm * 23).long(), 0, 23)
    # 假设weekday是归一化到0-1的原始范围0-6
    weekday = torch.clamp((weekday_norm * 6).long(), 0, 6)

    # 生成one-hot编码
    hour_onehot = F.one_hot(hour, num_classes=24).float()
    weekday_onehot = F.one_hot(weekday, num_classes=7).float()

    # 拼接成31维特征
    stdn_time_features = torch.cat([hour_onehot, weekday_onehot], dim=-1)

    return stdn_time_features

# 模型定义部分保持不变（使用你提供的Informer模型代码）
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
        # BiLSTM处理 - 增强序列建模
        lstm_out, _ = self.bilstm(x)  # [batch_size, seq_len, lstm_hidden*2]
        lstm_out = F.relu(self.dim_adapter(lstm_out))  # [batch_size, seq_len, d_model]

        x=x+lstm_out

        if self.distill_layers is not None:
            for i, layer in enumerate(self.encoder_layers):
                x = layer(x)
                if i < len(self.encoder_layers) - 1:  # 最后一层不蒸馏
                    x = self.distill_layers[i](x)
        else:
            for layer in self.encoder_layers:
                x = layer(x)

        # 残差连接：结合Informer和BiLSTM的输出
        x = x

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
            x = layer(x, enc_output, tgt_mask)

        # BiLSTM增强
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, lstm_hidden*2]
        lstm_out = F.relu(self.lstm_adapter(lstm_out))  # [batch_size, seq_len, d_model]

        # 残差连接
        x = x + lstm_out

        x = self.norm(x)

        # 只取最后一个时间步的输出，并投影到pred_len维度
        last_output = x[:, -1:, :]  # [batch_size, 1, d_model]

        # 投影到预测长度
        output = self.projection(last_output)  # [batch_size, 1, pred_len]
        output = output.squeeze(1)  # [batch_size, pred_len]

        return output


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)


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


class MultiScaleTemporalConv(nn.Module):
    """多尺度时间卷积模块"""

    def __init__(self, input_dim, hidden_dim, kernel_sizes=[2, 3, 6], dropout=0.1):
        super(MultiScaleTemporalConv, self).__init__()

        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            # 每个尺度使用因果卷积确保时序性
            padding = (kernel_size - 1)  # 因果卷积padding
            conv = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding, dilation=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=1),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv)

        # 特征融合层
        self.fusion = nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape

        # 转置以适配Conv1d: [batch_size, input_dim, seq_len]
        x_t = x.transpose(1, 2)

        # 多尺度卷积
        conv_outputs = []
        for conv in self.conv_layers:
            # 因果卷积，需要裁剪输出长度
            conv_out = conv(x_t)
            conv_out = conv_out[:, :, :seq_len]  # 裁剪到原始长度
            conv_out = conv_out.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
            conv_outputs.append(conv_out)

        # 拼接多尺度特征
        if len(conv_outputs) > 1:
            multi_scale_out = torch.cat(conv_outputs, dim=-1)
        else:
            multi_scale_out = conv_outputs[0]

        # 特征融合
        fused_features = self.fusion(multi_scale_out)
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)

        # 残差连接
        output = fused_features   # 确保维度匹配

        return output


# 修改MSTCNInformerWithBiLSTM类
class MSTCNInformerWithBiLSTM(nn.Module):
    """BiLSTM集成在编码器中的Informer模型，使用STDN风格的分解"""

    def __init__(self, enc_in: int, dec_in: int, time_feat_dim: int = 11,
                 c_out: int = 1, seq_len: int = 24, label_len: int = 12,
                 pred_len: int = 1, d_model: int = 512, n_heads: int = 8,
                 e_layers: int = 2, d_layers: int = 1, d_ff: int = 2048,
                 dropout: float = 0.1, lstm_hidden: int = 256,
                 lstm_layers: int = 2, mstcn_hidden=256,
                 mstcn_kernels=[2, 3, 6], use_stdn_decomposition=True,
                 use_onehot_time=False):  # 新增：是否使用one-hot时间特征
        super().__init__()

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.time_feat_dim = time_feat_dim
        self.use_stdn_decomposition = use_stdn_decomposition
        self.use_onehot_time = use_onehot_time

        # 如果使用one-hot，调整时间特征维度
        if use_onehot_time:
            effective_time_feat_dim = 31  # 24 + 7
        else:
            effective_time_feat_dim = time_feat_dim

        # STDN分解相关模块
        if use_stdn_decomposition:
            # 时空嵌入模块（使用有效的时间特征维度）
            self.st_embed = SpatioTemporalEmbed(
                time_feat_dim=effective_time_feat_dim,
                d_model=d_model
            )

            # STDN分解模块
            self.decomposition = STDN_TrendSeasonalDecomposition(d_model)

            # 趋势和季节的编码器
            self.trend_encoder = InformerEncoderWithBiLSTM(
                d_model=d_model,
                n_layers=e_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_distill=True,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers
            )

            self.seasonal_encoder = InformerEncoderWithBiLSTM(
                d_model=d_model,
                n_layers=e_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_distill=True,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers
            )

            # 可学习的融合参数（与STDN一致）
            self.fusion_param = nn.Parameter(torch.tensor(0.5))

            # 趋势和季节的MSTCN投影
            self.trend_mstcn_projection = nn.Linear(mstcn_hidden, d_model)
            self.seasonal_mstcn_projection = nn.Linear(mstcn_hidden, d_model)

            # 维度转换层（将趋势/季节特征转回原始维度供MSTCN使用）
            self.trend_to_original = nn.Linear(d_model, enc_in)
            self.seasonal_to_original = nn.Linear(d_model, enc_in)

        else:
            # 原始编码器（如果不使用分解）
            self.encoder = InformerEncoderWithBiLSTM(
                d_model=d_model,
                n_layers=e_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_distill=True,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers
            )

        # 编码器输入嵌入
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        # 解码器输入嵌入
        self.dec_embedding = nn.Linear(dec_in, d_model)

        # 解码器
        self.decoder = LSTMDecoder(
            d_model=d_model,
            pred_len=pred_len,
            n_layers=d_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers
        )

        # 多尺度时间卷积
        self.mstcn = MultiScaleTemporalConv(
            input_dim=enc_in,
            hidden_dim=mstcn_hidden,
            kernel_sizes=mstcn_kernels,
            dropout=dropout
        )

        # MSTCN特征到d_model的投影（用于非分解模式）
        self.mstcn_projection = nn.Linear(mstcn_hidden, d_model)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc: [batch_size, seq_len, enc_in] - 编码器输入序列
            x_mark_enc: [batch_size, seq_len, time_feat_dim] - 编码器时间特征
            x_dec: [batch_size, label_len + pred_len, dec_in] - 解码器输入序列
            x_mark_dec: [batch_size, label_len + pred_len, time_feat_dim] - 解码器时间特征
        Returns:
            output: [batch_size, pred_len] - 预测结果
        """

        # 如果需要，转换时间特征格式
        if self.use_onehot_time:
            x_mark_enc = convert_to_stdn_onehot(x_mark_enc)
            # 注意：x_mark_dec也需要转换，如果解码器也使用时间特征

        if self.use_stdn_decomposition:
            # STDN分解流程
            # 1. 投影输入
            x_enc_proj = self.enc_embedding(x_enc)  # [B, T, d_model]

            # 2. 生成时空嵌入M
            M = self.st_embed(x_mark_enc)  # [B, T, d_model]

            # 3. 使用M进行STDN分解
            trend, seasonal = self.decomposition(x_enc_proj, M)

            # 4. 将趋势和季节特征转换回原始维度供MSTCN使用
            trend_original = self.trend_to_original(trend)  # [B, T, enc_in]
            seasonal_original = self.seasonal_to_original(seasonal)  # [B, T, enc_in]

            # 5. MSTCN提取特征（分别处理趋势和季节）
            trend_mstcn_features = self.mstcn(trend_original)
            seasonal_mstcn_features = self.mstcn(seasonal_original)

            # 6. 投影回d_model维度
            trend_mstcn_projected = self.trend_mstcn_projection(trend_mstcn_features)
            seasonal_mstcn_projected = self.seasonal_mstcn_projection(seasonal_mstcn_features)

            # 7. 趋势和季节分别编码
            # 趋势编码器输入 = 趋势 + MSTCN特征
            trend_enc_input = trend + trend_mstcn_projected
            trend_enc_input = self.pos_encoding(trend_enc_input)
            trend_enc_input = self.dropout(trend_enc_input)
            trend_enc_out = self.trend_encoder(trend_enc_input)

            # 季节编码器输入 = 季节 + MSTCN特征
            seasonal_enc_input = seasonal + seasonal_mstcn_projected
            seasonal_enc_input = self.pos_encoding(seasonal_enc_input)
            seasonal_enc_input = self.dropout(seasonal_enc_input)
            seasonal_enc_out = self.seasonal_encoder(seasonal_enc_input)

            # 8. 融合趋势和季节特征
            fusion_weight = torch.sigmoid(self.fusion_param)
            enc_out = fusion_weight * trend_enc_out + (1 - fusion_weight) * seasonal_enc_out

        else:
            # 原始流程（不使用分解）
            # 1. 多尺度时间卷积提取先验特征
            mstcn_features = self.mstcn(x_enc)

            # 2. 投影到d_model维度
            mstcn_projected = self.mstcn_projection(mstcn_features)

            # 编码器部分
            enc_out = self.enc_embedding(x_enc) + mstcn_projected
            enc_out = self.pos_encoding(enc_out)
            enc_out = self.dropout(enc_out)
            enc_out = self.encoder(enc_out)

        # 解码器部分
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.pos_encoding(dec_out)
        dec_out = self.dropout(dec_out)

        # 解码器预测
        output = self.decoder(dec_out, enc_out)

        return output


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


# 评估函数
def evaluate_model(model, data_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y in data_loader:
            batch_x_enc = batch_x_enc.to(device)
            batch_x_mark_enc = batch_x_mark_enc.to(device)
            batch_x_dec = batch_x_dec.to(device)
            batch_x_mark_dec = batch_x_mark_dec.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return avg_loss, all_predictions, all_targets


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
TIME_FEAT_DIM = X_mark_enc_train.shape[2]  # 从数据中获取时间特征维度，应为11
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

# 模型初始化
# 在模型初始化时添加use_decomposition参数
model = MSTCNInformerWithBiLSTM(
    enc_in=enc_in,
    dec_in=dec_in,
    time_feat_dim=TIME_FEAT_DIM,  # 传入时间特征维度
    c_out=1,
    seq_len=SEQ_LENGTH,
    label_len=LABEL_LENGTH,
    pred_len=1,
    d_model=128,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=512,
    lstm_hidden=128,
    lstm_layers=2,
    dropout=0.2,
    mstcn_hidden=128,
    mstcn_kernels=[6,12,24],
    use_stdn_decomposition=True,  # 启用STDN分解
    use_onehot_time=False  # 使用原始时间特征，不进行one-hot转换
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# --- 训练循环 ---
best_loss = float('inf')
train_loss_history = []
val_loss_history = []
lr_history = []
patience = 10
patience_counter = 0

print("开始训练...")
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    epoch_train_loss = 0
    for batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y in train_loader:
        # 移动数据到设备
        batch_x_enc = batch_x_enc.to(device)
        batch_x_mark_enc = batch_x_mark_enc.to(device)
        batch_x_dec = batch_x_dec.to(device)
        batch_x_mark_dec = batch_x_mark_dec.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
        loss = criterion(outputs, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # 评估验证集
    avg_val_loss, avg_predictions, avg_targets = evaluate_model(model, val_loader, criterion, device)
    val_loss_history.append(avg_val_loss)

    # 记录学习率
    current_lr = scheduler.get_last_lr()[0]
    lr_history.append(current_lr)
    scheduler.step()

    # 改进的早停机制
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), 'D:\python-anaconda-learn1\交通流量\十二月\模型pth\MFjmstcnbilsinformer_model.pth')
        patience_counter = 0
        # print(f'Epoch {epoch + 1}: 最佳模型已保存，测试损失: {best_loss:.5f}')
    else:
        patience_counter += 1
        if patience_counter >= patience and epoch > 90:
            print(f"早停于第 {epoch + 1} 轮")
            break

    # 每5个epoch打印一次详细指标
    #if (epoch + 1) % 5 == 0:
    print(f'Epoch {epoch + 1}/{EPOCHS}, '
          f'Train Loss: {avg_train_loss:.5f}, '
          f'Val Loss: {avg_val_loss:.5f}, '
          f'LR: {current_lr:.6f}')

# --- 全面评估模块 ---
print("加载最佳模型进行最终评估...")
model.load_state_dict(torch.load('D:\python-anaconda-learn1\交通流量\十二月\模型pth\MFjmstcnbilsinformer_model.pth'))
model.eval()

# 收集所有预测结果
all_predictions = []
all_targets = []
# 评估测试集
test_loss, y_pred_scaled, y_true_scaled = evaluate_model(model, test_loader, criterion, device)
with torch.no_grad():
    for batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y in test_loader:
        batch_x_enc = batch_x_enc.to(device)
        batch_x_mark_enc = batch_x_mark_enc.to(device)
        batch_x_dec = batch_x_dec.to(device)
        batch_x_mark_dec = batch_x_mark_dec.to(device)

        predictions = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch_y.numpy())

# 合并预测结果
y_pred = np.concatenate(all_predictions, axis=0)
y_true = np.concatenate(all_targets, axis=0)


# 反归一化处理
def inverse_transform(scaler, pred):
    dummy = np.zeros((len(pred), 1))
    dummy[:, 0] = pred.ravel()
    return scaler.inverse_transform(dummy).ravel()


y_true_original = inverse_transform(scaler_target, y_true)
y_pred_original = inverse_transform(scaler_target, y_pred)

# 计算多种评估指标
mse = mean_squared_error(y_true_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_original, y_pred_original)
r2 = r2_score(y_true_original, y_pred_original)
mape = mean_absolute_percentage_error(y_true_original, y_pred_original)
smape = smape(y_true_original, y_pred_original)
mase = mase(y_true_original, y_pred_original, train_traffic, seasonality=24)

print(f"\n=== Informer模型测试集全面评估结果 ===")
print(f"测试集损失: {test_loss:.6f}")
print(f"MSE: {mse:.5f}")
print(f"RMSE: {rmse:.5f}")
print(f"MAE: {mae:.5f}")
print(f"R²: {r2:.5f}")
print(f"MAPE: {mape:.5f}%")
print(f"SMAPE: {smape:.5f}%")
print(f"MASE: {mase:.5f}")

# 输出预测统计信息
print(f"\n=== 预测统计 ===")
print(f"真实值范围: {y_true_original.min():.2f} - {y_true_original.max():.2f}")
print(f"预测值范围: {y_pred_original.min():.2f} - {y_pred_original.max():.2f}")
print(f"平均真实值: {y_true_original.mean():.2f}")
print(f"平均预测值: {y_pred_original.mean():.2f}")

# 可视化模块
plt.figure(figsize=(15, 12))

# 训练和验证损失曲线
plt.subplot(2, 2, 1)
plt.plot(train_loss_history, label='训练损失')
plt.plot(val_loss_history, label='验证损失')
plt.title('训练和验证损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 学习率变化曲线
plt.subplot(2, 2, 2)
plt.plot(lr_history)
plt.title('学习率变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True, alpha=0.3)

# 预测对比图
plt.subplot(2, 2, 3)
plt.plot(y_true_original[:200], label='真实值', linewidth=1, alpha=0.8)
plt.plot(y_pred_original[:200], label='预测值', linestyle='--', alpha=0.8)
plt.title('交通流量预测对比 (前200个样本)')
plt.legend()
plt.grid(True, alpha=0.3)

# 误差分布图
plt.subplot(2, 2, 4)
errors = y_pred_original - y_true_original
plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
plt.title('预测误差分布')
plt.xlabel('误差')
plt.ylabel('频次')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:\python-anaconda-learn1\交通流量\zzfinal\图表\MFjinformer_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存评估结果到文件
results_df = pd.DataFrame({
    'True_Value': y_true_original,
    'Predicted_Value': y_pred_original,
    'Error': errors
})
results_df.to_csv('D:\python-anaconda-learn1\交通流量\zzfinal\图表\MFjinformer_prediction_results.csv', index=False)

# 保存损失历史
loss_df = pd.DataFrame({
    'Epoch': range(1, len(train_loss_history) + 1),
    'Train_Loss': train_loss_history,
    'Val_Loss': val_loss_history,
    'Learning_Rate': lr_history
})
loss_df.to_csv('D:\python-anaconda-learn1\交通流量\zzfinal\图表\MFjinformer_training_history.csv', index=False)

print("\n评估结果已保存到 'MFjinformer_prediction_results.csv'")
print("训练历史已保存到 'MFjinformer_training_history.csv'")
print("图表已保存到 'MFjinformer_evaluation.png'")

# 模型架构总结
print("\n=== 模型架构总结 ===")
print(f"编码器输入维度: {enc_in}")
print(f"解码器输入维度: {dec_in}")
print(f"序列长度: {SEQ_LENGTH}")
print(f"标签长度: {LABEL_LENGTH}")
print(f"预测长度: {PRED_LEN}")
print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")

