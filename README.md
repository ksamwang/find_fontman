# find_fontman

图片字体识别工具 MVP。用户上传一张包含文字的图片，手动框选目标文字区域，系统用本地 OCR 辅助识别文本，再遍历本地字体库渲染同样文本并评分，返回最相似的 Top10 字体。

## 当前能力

- Go Web 服务：上传图片、框选区域、调用视觉服务、展示 Top10。
- Python 视觉服务：字体索引、裁剪区域、PaddleOCR 接入、Pillow/NumPy 渲染评分。
- 本地字体库：默认读取 `fonts/1中文简体`、`fonts/2中文繁体`、`fonts/4英文`。
- 评分拆解：`ssim`、`iou`、`edge`、`shape` 和总分。

> 当前仓库保留了 `proto/font_match.proto` 作为 gRPC 契约；由于本机未安装 `protoc`，MVP 运行路径先使用 Go 到 Python 的本地 HTTP JSON 接口。安装代码生成工具后可以按该 proto 替换传输层。

## 环境

建议使用 Python 3.11。Python 3.13 对 PaddleOCR/PaddlePaddle 的兼容风险较高。

```powershell
winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wheel
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r python_service\requirements.txt
```

如果暂时没有安装 PaddleOCR，也可以启动应用并手动输入文本；匹配字体需要至少安装 `pillow` 和 `numpy`。

## 启动

```powershell
go run .\cmd\findfontman
```

打开：

```text
http://localhost:8080
```

Go 服务会尝试拉起：

```text
python_service/service.py --addr 127.0.0.1:9091
```

可用环境变量：

- `ADDR`：Go Web 地址，默认 `:8080`
- `VISION_ADDR`：Python 服务地址，默认 `127.0.0.1:9091`
- `PYTHON`：指定 Python 解释器
- `SKIP_PYTHON_SERVICE=1`：不自动拉起 Python 服务
- `FONTMAN_MAX_CANDIDATES`：单次匹配最多粗排候选数，默认 `200`

## 使用流程

1. 点击上传图片。
2. 在图片上拖拽框选一段文字。
3. 点击 `OCR 识别`，或直接在文本框手动输入/修正文字。
4. 点击 `匹配字体`。
5. 查看 Top10 字体、预览图和评分拆解。

## 数据目录

运行时生成文件位于 `data/`：

- `data/uploads/`：上传图片
- `data/previews/`：字体渲染预览图
- `data/crops/`：OCR 裁剪图
- `data/font_index.sqlite`：字体索引缓存

`fonts/` 和 `data/` 默认不纳入 Git。
