# Entropia Riko

[English](README.md) | 中文

Entropia Riko 是一个专业的**节点图深度学习编辑器**——ComfyUI 风格的可视化工作流，
支持 PyTorch（及可选 TensorFlow/Keras），带 Blender 式模块化工作区、实时训练曲线、
代码导出、插件系统和内置文件管理器。

它既能作为 **Web 应用**（浏览器）运行，也内置 **Electron 壳**，可当**独立桌面应用**
使用——两种方式任选。

## 特性

- **200+ 节点**：数学、张量操作、神经网络层/激活、注意力、归一化、归约、形状、einsum、
  损失、数据加载、模型推理、子图引用、Hugging Face（Diffusers / Transformers）、
  TF/Keras 对应节点。
- **节点画布**（React Flow）：右键搜索菜单、拖拽连线、节点卡片实时输出预览。
- **Blender 式模块化工作区**：拖角分离/合并/缩放任意面板（分裂时两个圆角矩形蓝色预览）、
  任意切换窗口类型、多个工作区标签（Layout / Code / Training / MNIST Studio /
  Text→Image …）。
- **代码编辑器**：记事本风格（File/Edit 菜单 + 工具栏：新建/打开/保存/撤销/重做/
  剪切/复制/粘贴），用于预览/编辑导出的 PyTorch 代码。
- **训练 + 实时 Loss 曲线**：SSE 流式逐步回传损失到 SVG 图表。
- **干净代码导出**：PyTorch `nn.Module` 与 TensorFlow `tf.keras.Model`。
- **Asset Library 与 New File**：工作目录文件管理器（文件夹拖拽、右键新建/重命名/删除、
  「展开完整节点」把文件图内联进画布而非子图引用、每个文件的 PyTorch 代码预览）。
- **内置文件资源管理器**：Windows 式导入/导出（浏览、前进/后退、快速访问、最近文件夹；
  复制文件/文件夹，而非浏览器直接下载）。
- **插件系统**：从 `.py` 加载插件、开关启用/禁用；工作区面板与 Preferences 都能管理。
- **手写板**：画 28×28 数字 → 生成 `constant` 节点喂 MNIST 示例推理。
- **主题**：Light / Dark / System / **Liquid Glass**（苹果风格半透明玻璃）。
- **可脱出浮动窗口**：所有对话框都是可拖拽窗口。
- **`.rik` 二进制 + `.riko` ASCII 格式**，含完整 metadata/settings。

## 快速开始（浏览器）

```bash
cd entropia-riko
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
npm install

# 终端 1 — API (http://localhost:8000)
.venv/bin/python -m uvicorn src.server.app:app --reload --port 8000

# 终端 2 — 前端 (http://localhost:5173)
npm run dev
```

打开 **http://localhost:5173**（`/api` 走 Vite 代理到 :8000）。

## 快速开始（桌面应用）

Electron 壳会启动后端，并在 Vite 开发服务器上打开原生窗口：

```bash
npm install --save-dev electron
npm run dev &                # 保持 Vite 开发服务器运行
npm run desktop
```

如需指向其他前端地址，可设置环境变量 `RIKO_DEV_URL`。

## .riko / .ric 文件格式

`.riko` 是可读 JSON；`.ric` 是同一文档在 `ERIK` 魔数头之后 zlib 压缩的二进制。两者都
包含 `version`、`metadata`（name、app、appVersion）、`nodes`、`edges`、`settings`
（主题、背景图）。详见 [`docs/FILE_FORMAT.md`](docs/FILE_FORMAT.md)。

## 插件

插件位于 `plugins/*/`（`plugin.json` 清单 + 通过 `@register` 注册节点的 `entry`
Python 模块）。可从 `.py` 文件加载更多，并在 **Plugins** 面板或 **Preferences →
Plugins** 里开关。禁用的插件会被跳过，其节点不被注册。内置示例：`example_plugin`、
`math_extra`、`stat_extra`。

## 开发命令

```bash
.venv/bin/python -m unittest discover -s tests -t .   # Python 测试
npm test                                              # 前端测试（vitest）
npm run build                                         # 类型检查 + 生产构建
npm run dev                                           # Vite 开发服务器
npm run desktop                                       # Electron 桌面壳
.venv/bin/python scripts/make_brand_assets.py         # 品牌素材助手（见脚本说明）
```

## 项目结构

```
src/
├── ui/         React 应用（画布、面板、代码编辑器、文件管理器…）
├── core/       Tensor IR + 图文档模型（.riko/.ric）
├── runtime/    注册表、执行器、PyTorch/TF 代码生成、训练器、子图
├── backend/    Torch 设备检测 + 转换
├── nodes/      节点定义
├── plugins/    插件加载器
└── server/     FastAPI API 服务器
plugins/        内置插件
examples/       开箱即用的预连接示例图（dataset → model → loss → output）
electron/       桌面壳（main + preload）
scripts/        品牌素材生成器
public/brand/   logo.svg + hero.jpg（原地替换即可换品牌）
```

## 文档

- **[用户指南](docs/USER_GUIDE.md)** — 完整手册（界面、节点、训练、导出、API）。
- `docs/`：`APP_SPEC.md`、`APP_ARCHITECTURE.md`、`API.md`、`NODE_SYSTEM.md`、
  `DATA_FORMAT.md`、`FILE_FORMAT.md`、`UI_STANDARD.md`、`TORCH_BACKEND.md`、
  `CROSS_PLATFORM.md`。

## 许可证

MIT
