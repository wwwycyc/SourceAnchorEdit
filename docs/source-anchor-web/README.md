# Source Anchor Web Demo

这是一个轻量的本地原型，目标是把以下链路串起来：

1. 上传图片并强制校验。
2. 前端自动把图片处理成 `512 x 512`。
3. 调用识图接口生成英文 `source prompt`。
4. 调用 prompt 接口把用户指令转成英文并组合成英文 `target prompt`。
5. 将请求提交到本地 `source_anchor` 后端并返回编辑结果图。

## 运行

在仓库根目录执行：

```powershell
python tools\source_anchor_web_demo.py
```

默认地址：

```text
http://127.0.0.1:8765
```

静态页面目录：

```text
docs/source-anchor-web/
```

## 当前行为

- `POST /api/vision/source-prompt`
  默认先调用 Mimo 接口生成英文 `source prompt`，失败时回退到 mock。
- `POST /api/prompt/compose-target`
  默认先调用 Mimo 接口生成英文 `translatedInstruction` 和 `targetPrompt`，失败时回退到规则模式。
- `POST /api/source-anchor/run`
  会把请求物化到 `runs/source_anchor_web_demo/jobs/<job_id>/`，生成：
  - `source_512.png`
  - `sample.json`
  - `request.json`
  - `experiment.json`

它会进一步调用新的 `source_anchor` 主线，生成真正的编辑结果图。

## 说明

- 如果页面是直接用文件方式打开，或者本地服务没有启动，前端会自动回退到浏览器内的 mock / 规则模式。
- 当前服务会优先读取环境变量 `MIMO_API_KEY`、`MIMO_BASE_URL`、`MIMO_MODEL`；未设置时使用本地默认配置。
- 要更换成别的 LLM / VLM API 时，只需要替换这几个接口的返回逻辑，不需要改页面交互层。
