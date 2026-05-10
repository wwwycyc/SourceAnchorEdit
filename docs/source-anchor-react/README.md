# Source Anchor React Demo

这是一个独立于现有 `docs/source-anchor-web/` 的 React 重写版本。

它不会改动当前 Python demo 的静态目录绑定，只是单独提供一个更好维护的前端项目，继续复用现有接口：

- `POST /api/vision/source-prompt`
- `POST /api/prompt/compose-target`
- `POST /api/source-anchor/run`

## 启动方式

先启动原来的 Python demo 服务：

```powershell
python tools\source_anchor_web_demo.py
```

再进入这个 React 项目目录安装依赖并启动：

```powershell
cd docs\source-anchor-react
npm install
npm run dev
```

默认开发地址：

```text
http://127.0.0.1:5176
```

## API 地址

开发环境默认通过 Vite 代理把 `/api/*` 转发到：

```text
http://127.0.0.1:8765
```

如果你想显式指定后端地址，可以新建 `.env.local`：

```text
VITE_API_BASE_URL=http://127.0.0.1:8765
```

## 说明

- 这个目录是新增项目，不会恢复或修改现有 `docs/source-anchor-web/`。
- 页面保留了原型链路：上传图片、前端压到 `512 x 512`、生成 `source prompt`、翻译编辑指令、调用 `source_anchor` 返回结果图。
- 如果后端接口不可用，并且页面里打开了“规则回退”，前两步会自动退回浏览器端 mock / 规则逻辑。
