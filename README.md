# Parastat

Parastat 是一个高性能的 CPU/GPU 并行数据统计分析库，旨在通过 NumPy/Pandas 和 CuPy/CuDF 加速核心矩阵运算，提供统计分析和机器学习任务的快速实现。支持的功能包括普通最小二乘法（OLS）、广义线性模型（GLM）、主成分分析（PCA）、奇异值分解（SVD）、Bootstrap 方法和蒙特卡洛模拟（Monte Carlo）。

## 功能亮点

- **高效计算**：利用 CPU（NumPy/Pandas）或 GPU（CuPy/CuDF）进行矩阵运算，显著提升性能。
- **核心统计方法**：
  - 普通最小二乘法（OLS）和广义线性模型（GLM）用于回归分析。
  - 主成分分析（PCA）和奇异值分解（SVD）用于数据降维和分解。
  - Bootstrap 和 Monte Carlo 方法用于统计推断和不确定性分析。
- **灵活的数据支持**：支持 Parquet 文件格式（通过 `pyarrow`），适合处理大数据集。
- **命令行工具**：提供 `parastat` 命令行接口，方便快速执行统计任务。

## 安装

### CPU 安装

在 Python 3.8 或以上版本的环境中，使用以下命令安装 Parastat：

```bash
pip install parastat
```

### GPU 安装（可选）

若需启用 GPU 加速，需安装 CuPy（支持 CUDA 12.x）：

```bash
pip install parastat[gpu]
```

**注意**：GPU 安装需要 NVIDIA GPU 和兼容的 CUDA 环境。确保已安装 CUDA Toolkit 和 CuPy。

### 开发安装

若需进行本地开发，克隆仓库并安装开发依赖：

```bash
git clone https://github.com/fept-2024/parastat.git
cd parastat
pip install -e .[dev]
```

开发依赖包括 `pytest`（测试）、`ruff`（代码格式化）和 `mypy`（类型检查）。

### Parquet 支持（可选）

若需处理 Parquet 文件，安装 `pyarrow`：

```bash
pip install parastat[parquet]
```

## 使用示例

以下是一些 Parastat 的基本用法示例：

### 1. 运行 OLS 回归

```python
import parastat
import numpy as np

# 生成示例数据
X = np.random.rand(100, 3)  # 100 行，3 个特征
y = np.random.rand(100)     # 目标变量

# 执行 OLS 回归
model = parastat.OLS()
model.fit(X, y)
print("系数:", model.coef_)
```

### 2. 主成分分析（PCA）

```python
import parastat
import pandas as pd

# 加载数据
data = pd.DataFrame(np.random.rand(100, 5))

# 执行 PCA
pca = parastat.PCA(n_components=2)
pca.fit(data)
transformed_data = pca.transform(data)
print("主成分:", transformed_data)
```

### 3. 通过命令行运行统计任务

```bash
parastat --task ols --input data.csv --output results.csv
```

更多命令行选项请运行：

```bash
parastat --help
```

## 依赖

- **核心依赖**：
  - `numpy>=1.22`
  - `pandas>=1.5`
- **可选依赖**：
  - `cupy-cuda12x`（GPU 支持）
  - `pyarrow>=9`（Parquet 文件支持）
- **开发依赖**：
  - `pytest>=7`
  - `ruff>=0.5`
  - `mypy>=1.0`

## 文档

更多详细信息请参考 项目文档（待完善）。包括：

- API 参考
- 高级用法（例如 GPU 加速设置）
- 性能优化建议

## 贡献

欢迎为 Parastat 贡献代码！请按照以下步骤：

1. 克隆仓库：

   ```bash
   git clone https://github.com/fept-2024/parastat.git
   ```

2. 创建特性分支：

   ```bash
   git checkout -b feature/your-feature
   ```

3. 提交更改并运行测试：

   ```bash
   pytest
   ruff check .
   mypy .
   ```

4. 推送并创建 Pull Request。

请遵循 贡献指南（待完善）。

## 许可证

Parastat 使用 MIT 许可证。详情请见 `LICENSE` 文件。

## 联系

- **问题反馈**：请在 GitHub 仓库提交 Issue