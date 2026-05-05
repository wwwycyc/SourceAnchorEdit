# Source Anchor

[English](README.md) | [中文](README_zh.md)

这是一个围绕最终版 `source anchor` 方法整理出来的独立开源仓库。

当前仓库只保留一条最终主线：

- source anchor
- 无时序累计
- 不引入弱提示信息
- ROI 始终启用
- ROI 来源支持 `live` 和 `cache`

核心实现位于 [src/sourceanchor](src/sourceanchor)。

## 主要入口

- 单样本运行：[scripts/run_single.py](scripts/run_single.py)
- 批量运行：[scripts/run_batch.py](scripts/run_batch.py)
- ROI cache 构建：[scripts/build_roi_cache.py](scripts/build_roi_cache.py)
- 数据集转换：[scripts/convert_dataset.py](scripts/convert_dataset.py)
- 本地可视化演示：[scripts/launch_web_demo.py](scripts/launch_web_demo.py)

## 核心文档

- 输入格式：[docs/input_format.md](docs/input_format.md)
- 配置说明：[docs/config.md](docs/config.md)
- 方法说明：[docs/method.md](docs/method.md)
- 复现说明：[docs/reproducibility.md](docs/reproducibility.md)

## 快速开始

运行最小样例：

```powershell
python scripts\run_single.py --config configs\experiments\source_anchor.demo.example.yaml
```

构建 ROI cache：

```powershell
python scripts\build_roi_cache.py --config configs\experiments\source_anchor.build_cache.example.yaml
```

使用 cache 运行：

```powershell
python scripts\run_single.py --config configs\experiments\source_anchor.use_cache.example.yaml
```

启动本地可视化页面：

```powershell
python scripts\launch_web_demo.py
```

## 当前已包含

- 分层配置加载
- 标准样本格式
- 单样本运行入口
- 批量运行入口
- ROI live/cache 工作流
- ROI cache 构建脚本
- PIE-Bench 数据集转换
- 本地 Web demo 后端
