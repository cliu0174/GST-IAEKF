# SOC 估计算法对比平台

基于 CALCE 电池数据集的锂电池 SOC (State of Charge) 估计算法实现与对比。

## 算法实现

| 算法 | 说明 |
|------|------|
| AEKF | 自适应扩展卡尔曼滤波 (Adaptive Extended Kalman Filter) |
| UKF | 无迹卡尔曼滤波 (Unscented Kalman Filter) |
| SR-UKF | 鲁棒平方根无迹卡尔曼滤波 (Robust Square-Root UKF) |
| GST-IAEKF | 广义强跟踪迭代自适应扩展卡尔曼滤波 |

## 项目结构

```
CALCE/
├── main.py                          # 主运行脚本 (单数据集)
├── algorithms/
│   ├── functions/                   # 算法实现
│   │   ├── aekf.py                 # AEKF 算法
│   │   ├── ukf.py                  # UKF 算法
│   │   ├── sr_ukf.py               # SR-UKF 算法
│   │   ├── gst_iaekf.py            # GST-IAEKF 算法
│   │   ├── battery_model.py        # 电池模型
│   │   └── ffrls.py                # 在线参数辨识
│   └── tools/
│       ├── compare_all_datasets.py # 一键对比所有数据集
│       ├── process_battery_data.py # 数据预处理
│       └── plot_dst_curves.py      # 绘制曲线工具
├── dataset/
│   ├── processed/                   # 预处理后的数据
│   │   ├── 25C_DST_80SOC.csv
│   │   ├── 25C_DST_50SOC.csv
│   │   ├── 25C_FUDS_80SOC.csv
│   │   └── ...
│   └── OCV-SOC/                     # OCV-SOC 曲线数据
└── results/
    └── graphs/                      # 输出结果图
```

## 使用方法

### 1. 运行单个数据集

编辑 `main.py` 顶部的配置区域：

```python
# 选择模型 (None = 所有模型)
RUN_MODELS = None                    # 运行所有模型
RUN_MODELS = ["AEKF"]               # 只运行 AEKF
RUN_MODELS = ["AEKF", "UKF"]        # 运行 AEKF 和 UKF

# 选择数据集
DATA_FILE = "25C_DST_80SOC.csv"

# SOC 过滤范围
SOC_MIN = 10.0
SOC_MAX = 100.0
```

运行：
```bash
python main.py
```

输出目录: `results/graphs/{数据集名}/`

### 2. 一键对比所有数据集

编辑 `algorithms/tools/compare_all_datasets.py` 顶部的配置区域：

```python
# 选择模型 (None = 所有模型)
RUN_MODELS = None

# 选择数据集 (None = 所有数据集)
RUN_DATASETS = None
```

运行：
```bash
python algorithms/tools/compare_all_datasets.py
```

输出目录: `results/graphs/summary/`
- `comparison_bar.png` - 柱状对比图
- `comparison_heatmap.png` - RMSE 热力图
- `comparison_results.csv` - CSV 结果表

## 可用数据集

| 数据集 | 工况 | 初始 SOC |
|--------|------|----------|
| 25C_DST_80SOC.csv | DST | 80% |
| 25C_DST_50SOC.csv | DST | 50% |
| 25C_FUDS_80SOC.csv | FUDS | 80% |
| 25C_FUDS_50SOC.csv | FUDS | 50% |
| 25C_US06_80SOC.csv | US06 | 80% |
| 25C_US06_50SOC.csv | US06 | 50% |
| 25C_BBDST_80SOC.csv | BBDST | 80% |
| 25C_BBDST_50SOC.csv | BBDST | 50% |

## 评估指标

- **RMSE**: 均方根误差 (Root Mean Square Error)
- **MAE**: 平均绝对误差 (Mean Absolute Error)
- **Max Error**: 最大绝对误差

## 依赖

```
numpy
pandas
matplotlib
scipy
```

安装：
```bash
pip install numpy pandas matplotlib scipy
```
