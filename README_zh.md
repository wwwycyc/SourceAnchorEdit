# Source Anchor

[English](README.md) | [中文](README_zh.md)

基于扩散模型的高保真图像编辑方法。

## 特性

- 高保真度编辑：保持非编辑区域不变
- 源锚定机制：确保背景结构稳定
- 动态掩码：精准控制编辑区域
- 支持 ROI 缓存加速

## 环境配置

### 1. 安装依赖

```bash
# 安装 PyTorch（根据你的 CUDA 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
```

或者使用可编辑模式安装：

```bash
pip install -e .
```

### 2. 配置模型路径

复制并编辑配置文件：

```bash
cp configs/models/local_models.example.yaml configs/models/local_models.local.yaml
```

编辑 `local_models.local.yaml`，设置模型路径：

```yaml
models:
  sd_model: runwayml/stable-diffusion-v1-5  # 或本地路径
  clip_model: openai/clip-vit-large-patch14  # 或本地路径
  dino_weights: null  # 可选
```

## 快速开始

### 单样本编辑

```bash
python scripts/run_single.py --config configs/experiments/source_anchor.demo.example.yaml
```

### 批量处理

```bash
python scripts/run_batch.py --config configs/experiments/source_anchor.use_cache.example.yaml
```

### 构建 ROI 缓存（可选，加速重复实验）

```bash
python scripts/build_roi_cache.py --config configs/experiments/source_anchor.build_cache.example.yaml
```

### 启动 Web 演示

```bash
python scripts/launch_web_demo.py
```

然后在浏览器中打开显示的地址。

## 输入格式

创建 `sample.json`：

```json
{
  "sample_id": "example_001",
  "source_image_path": "path/to/source.png",
  "source_prompt": "a cat sitting on a chair",
  "target_prompt": "a dog sitting on a chair",
  "editing_region": "auto"
}
```

详细格式说明见 [docs/input_format.md](docs/input_format.md)。

## 输出结果

运行后，结果保存在 `runs/` 目录：

```
runs/
  source_anchor_<timestamp>/
    samples/
      <sample_id>/
        source.png              # 原图
        edited.png              # 编辑结果
        source_reconstruction.png  # 重建图像
        roi_soft.png            # ROI 可视化（软掩码）
        roi_hard.png            # ROI 可视化（硬掩码）
        overview.png            # 完整对比图
        debug.json              # 调试信息
```

## 配置说明

配置文件采用分层结构：

- `configs/models/` - 模型路径配置
- `configs/methods/` - 方法参数配置
- `configs/experiments/` - 实验配置

详细说明见 [docs/config.md](docs/config.md)。

## 项目结构

```
source_anchor_release/
├── configs/              # 配置文件
├── docs/                 # 文档
├── examples/             # 示例数据
├── scripts/              # 运行脚本
├── src/sourceanchor/     # 核心代码
│   ├── inversion/        # 图像反演
│   ├── method/           # 核心算法
│   ├── roi/              # ROI 生成
│   └── runtime/          # 运行时组件
└── tools/                # 辅助工具
```

## 常见问题

### 1. 如何使用本地模型？

在 `local_models.local.yaml` 中设置本地路径：

```yaml
models:
  sd_model: /path/to/stable-diffusion-v1-5
  clip_model: /path/to/clip-vit-large-patch14
```

### 2. 如何调整编辑强度？

在方法配置中调整 `guidance_scale`：

```yaml
method:
  guidance_scale: 7.5  # 默认值，增大则编辑更强
```

### 3. GPU 内存不足怎么办？

启用内存优化选项：

```yaml
runtime:
  attention_slicing: true
  vae_slicing: true
  enable_cpu_offload: true  # 最激进的选项
```

## 许可证

见 [LICENSE](LICENSE) 文件。

## 相关文档

- [输入格式说明](docs/input_format.md)
- [配置详解](docs/config.md)
- [方法说明](docs/method.md)
- [复现指南](docs/reproducibility.md)
