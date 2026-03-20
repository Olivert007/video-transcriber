# video-transcriber

基于 OpenAI Whisper 的本地视频/音频转录工具。将视频或音频文件转录为带时间戳的文本，支持多语言自动检测、多种输出格式，全程本地 GPU 推理，无需联网、无需付费。

## 环境要求

- Python 3.10+
- CUDA GPU（推荐，CPU 也可运行但较慢）
- ffmpeg（用于从视频中提取音频）

## 安装

```bash
# 1. 创建或激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install openai-whisper

# 3. 安装 ffmpeg（如果系统中没有）
# Ubuntu/Debian:  sudo apt install ffmpeg
# macOS:          brew install ffmpeg
# 或下载静态版:   https://johnvansickle.com/ffmpeg/
```

> **关于 Whisper 模型**：首次使用某个模型时会自动下载，之后直接使用本地缓存。模型完全免费开源（MIT 许可证）。
>
> 模型缓存路径按以下优先级确定：
> 1. `--model-dir` 命令行参数
> 2. `WHISPER_CACHE_DIR` 环境变量
> 3. 默认路径 `~/.cache/whisper/`
>
> 在公用服务器上建议通过 `--model-dir` 或环境变量将模型存放到磁盘空间充足的位置。本项目内已预留目录 **`whisper_models/`** 用于存放模型，可直接指定该目录，避免占用 `~/.cache`。

## 快速开始

```bash
# 最简单 — 自动检测语言，large 模型，输出 Markdown
python transcribe.py video.mp4

# 指定中文转录
python transcribe.py video.mp4 --language zh

# 翻译为英文（适合中英混合内容）
python transcribe.py video.mp4 --translate

# 先英文转录，再翻译为中文（推荐用于英文技术视频）
python transcribe.py video.mp4 --language en --translate-to zh-CN

# 输出 SRT 字幕
python transcribe.py video.mp4 --format srt

# 使用项目内 whisper_models 目录（推荐，命名清晰）
python transcribe.py video.mp4 --model-dir ./whisper_models

# 或通过环境变量设置（一次设置，所有调用生效）
export WHISPER_CACHE_DIR=./whisper_models
python transcribe.py video.mp4
```

## 输出格式

| 格式 | 参数 | 说明 |
|------|------|------|
| Markdown | `--format md` | **默认**，按约 60 秒分段，带时间戳 |
| SRT 字幕 | `--format srt` | 标准字幕文件，可直接导入视频播放器 |
| JSON | `--format json` | 结构化数据，方便程序化处理 |
| 纯文本 | `--format txt` | 无时间戳，纯文字 |

## 完整参数

```
python transcribe.py <输入文件> [选项]

位置参数:
  input                    输入的视频或音频文件

选项:
  -m, --model {tiny,base,small,medium,large}
                           Whisper 模型 (默认: large)
  -l, --language LANGUAGE  强制语言代码, 如 zh/en/ja (默认: 自动检测)
  -t, --translate          翻译为英文
  --translate-to TARGET    二次翻译目标语言代码, 如 zh-CN/en/ja（先转录再翻译）
  -f, --format {txt,srt,md,json}
                           输出格式 (默认: md)
  -o, --output OUTPUT      输出文件路径 (默认: <输入文件名>_transcription.<格式>)
  -g, --gpu GPU            GPU 编号, -1 为 CPU (默认: 0)
  -p, --prompt PROMPT      初始提示词, 引导模型使用特定术语或风格
  --model-dir MODEL_DIR    模型下载/缓存目录 (默认: $WHISPER_CACHE_DIR 或 ~/.cache/whisper)
```

## 使用示例

```bash
# 用 medium 模型（速度快约 3 倍，质量略低）
python transcribe.py lecture.mp4 --model medium

# 指定 GPU 2
python transcribe.py interview.mp4 --gpu 2

# CPU 模式（无 GPU 时）
python transcribe.py podcast.mp3 --gpu -1

# 带提示词（引导专业术语）
python transcribe.py meeting.mp4 --prompt "以下是关于 ROCm 和 AMD GPU 的技术讨论"

# 英文视频输出中文稿（保留时间戳分段）
python transcribe.py talk.mp4 --language en --translate-to zh-CN --format md

# 指定输出路径
python transcribe.py talk.mp4 --output ~/documents/talk_subtitle.srt --format srt
```

> 使用 `--translate-to` 需要额外安装依赖：
> `pip install deep-translator`

## 模型选择参考

| 模型 | 参数量 | 磁盘占用 | 显存需求 | 相对速度 | 推荐场景 |
|------|--------|----------|----------|----------|----------|
| tiny | 39M | ~75 MB | ~1 GB | ★★★★★ | 快速预览、测试 |
| base | 74M | ~140 MB | ~1 GB | ★★★★☆ | 简单内容、对质量要求不高 |
| small | 244M | ~460 MB | ~2 GB | ★★★☆☆ | 一般用途、较均衡 |
| medium | 769M | 1.5 GB | ~5 GB | ★★☆☆☆ | **推荐**，质量与速度兼顾 |
| large | 1.5B | 2.9 GB | ~10 GB | ★☆☆☆☆ | 最高质量、多语言混合内容 |

## 支持的输入格式

- **视频**: mp4, mkv, avi, mov, webm, flv, ts ...
- **音频**: wav, mp3, m4a, flac, ogg, aac, wma ...

视频文件会自动用 ffmpeg 提取音频（16kHz 单声道 WAV），无需手动转换。

## 项目结构

```
video-transcriber/
├── README.md
├── transcribe.py
├── remove_timestamps.py
└── whisper_models/     # 模型缓存目录（可选，首次指定后自动下载到此，见 .gitignore）
```

将模型下载到项目内 **`whisper_models/`** 目录（命名清晰、便于管理）：

```bash
# 在 video-transcriber 目录下执行，模型会下载到 ./whisper_models
python transcribe.py your_video.mp4 --model-dir ./whisper_models
```

## 模型缓存管理

### 自定义缓存路径

默认模型缓存在 `~/.cache/whisper/`。可使用项目内 `whisper_models/` 或任意自定义路径：

```bash
# 项目内目录（推荐）
python transcribe.py video.mp4 --model-dir ./whisper_models

# 其他自定义路径
python transcribe.py video.mp4 --model-dir /data/whisper_models

# 环境变量（推荐写入 .bashrc，长期生效）
export WHISPER_CACHE_DIR=/path/to/whisper_models
python transcribe.py video.mp4
```

路径优先级：`--model-dir` > `WHISPER_CACHE_DIR` 环境变量 > `~/.cache/whisper`

### 清理缓存

```bash
# 查看缓存大小（替换为实际缓存路径）
du -sh ~/.cache/whisper/

# 删除所有缓存模型
rm -rf ~/.cache/whisper/
```
