#!/usr/bin/env python3
"""
视频/音频转录工具 —— 将本地视频或音频文件转录为带时间戳的文本。

依赖：
    pip install openai-whisper
    pip install deep-translator   # 仅 --translate-to 需要
    需要 ffmpeg 在 PATH 中

用法：
    python transcribe.py video.mp4
    python transcribe.py audio.wav --model large --language zh
    python transcribe.py video.mp4 --output result.md --format md
    python transcribe.py video.mp4 --translate    # 全部翻译为英文
    python transcribe.py video.mp4 --language en --translate-to zh-CN
    python transcribe.py video.mp4 --gpu 2        # 指定 GPU
    python transcribe.py video.mp4 --model-dir /data/whisper_models  # 自定义模型目录

模型存储路径优先级: --model-dir 参数 > 环境变量 WHISPER_CACHE_DIR > 默认 ~/.cache/whisper

支持的输入格式：mp4, mkv, avi, mov, webm, flv, m4a, mp3, wav, flac, ogg 等
输出格式：txt（纯文本）、srt（字幕）、md（Markdown 带时间戳）、json（结构化数据）
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def format_timestamp(seconds: float, style: str = "srt") -> str:
    """将秒数转为时间戳字符串。"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if style == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def extract_audio(input_path: str, output_path: str) -> None:
    """用 ffmpeg 从视频中提取音频，转为 16kHz 单声道 WAV。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[错误] ffmpeg 提取音频失败:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def probe_duration(input_path: str) -> float | None:
    """获取媒体文件时长（秒）。"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return None


def is_audio_file(path: str) -> bool:
    """判断文件是否为纯音频格式。"""
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    return Path(path).suffix.lower() in audio_exts


def resolve_model_dir(model_dir: str | None = None) -> str | None:
    """按优先级确定模型目录: 参数 > 环境变量 > None（使用 whisper 默认路径）。"""
    if model_dir:
        return model_dir
    env_dir = os.environ.get("WHISPER_CACHE_DIR")
    if env_dir:
        return env_dir
    return None


def transcribe(
    input_path: str,
    model_name: str = "large",
    language: str | None = None,
    translate: bool = False,
    gpu: int = 0,
    initial_prompt: str | None = None,
    model_dir: str | None = None,
) -> dict:
    """
    执行转录，返回 whisper 结果字典。

    参数:
        input_path: 输入文件路径
        model_name: whisper 模型 (tiny/base/small/medium/large)
        language: 强制语言代码, None 则自动检测
        translate: 是否翻译为英文
        gpu: 使用的 GPU 编号, -1 表示 CPU
        initial_prompt: 初始提示词, 可引导风格和用语
        model_dir: 模型下载/缓存目录, None 则按优先级回退
    """
    import whisper

    download_root = resolve_model_dir(model_dir)
    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    cache_display = download_root or "~/.cache/whisper"
    print(f"[1/3] 加载模型 {model_name} → {device} (缓存: {cache_display}) ...")
    t0 = time.time()
    model = whisper.load_model(model_name, device=device, download_root=download_root)
    print(f"       模型加载完成 ({time.time() - t0:.1f}s)")

    # 如果是视频文件，先提取音频
    audio_path = input_path
    tmp_wav = None
    if not is_audio_file(input_path):
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        audio_path = tmp_wav.name
        print(f"[2/3] 提取音频中 ...")
        extract_audio(input_path, audio_path)
        print(f"       音频提取完成 → {audio_path}")
    else:
        print(f"[2/3] 输入为音频文件，跳过提取")

    duration = probe_duration(audio_path)
    if duration:
        m, s = divmod(int(duration), 60)
        h, m = divmod(m, 60)
        print(f"       时长: {h:02d}:{m:02d}:{s:02d}")

    print(f"[3/3] 转录中 (language={language or 'auto'}, translate={translate}) ...")
    t1 = time.time()

    task = "translate" if translate else "transcribe"
    transcribe_opts = dict(
        task=task,
        verbose=False,
    )
    if language:
        transcribe_opts["language"] = language
    if initial_prompt:
        transcribe_opts["initial_prompt"] = initial_prompt

    result = model.transcribe(audio_path, **transcribe_opts)

    elapsed = time.time() - t1
    detected_lang = result.get("language", "unknown")
    n_segments = len(result.get("segments", []))
    print(f"       转录完成 ({elapsed:.1f}s, 检测语言={detected_lang}, {n_segments} 段)")

    if tmp_wav:
        os.unlink(tmp_wav.name)

    return result


def split_text_by_words(text: str, max_chars: int = 4000) -> list[str]:
    """将较长文本按词切分，避免单次翻译长度超限。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    if not words:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    chunks = []
    current_words = []
    current_len = 0

    for word in words:
        word_len = len(word)
        sep_len = 1 if current_words else 0
        if current_words and current_len + sep_len + word_len > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = word_len
        elif word_len > max_chars:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_len = 0
            chunks.extend(word[i:i + max_chars] for i in range(0, word_len, max_chars))
        else:
            current_words.append(word)
            current_len += sep_len + word_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def translate_with_retry(translator, text: str, retries: int = 3) -> str:
    """带重试的单段翻译。"""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return translator.translate(text)
        except Exception as err:  # pragma: no cover - 网络异常分支
            last_error = err
            if attempt < retries:
                time.sleep(min(3, attempt))
    raise RuntimeError(f"翻译失败: {last_error}")


def translate_result_segments(
    result: dict,
    target_language: str,
    source_language: str = "auto",
) -> dict:
    """
    将转录结果按 segment 翻译为目标语言，并保留时间戳信息。
    依赖: pip install deep-translator
    """
    try:
        deep_translator = importlib.import_module("deep_translator")
        GoogleTranslator = getattr(deep_translator, "GoogleTranslator")
    except Exception:
        print(
            "[错误] 使用 --translate-to 需要安装 deep-translator: pip install deep-translator",
            file=sys.stderr,
        )
        sys.exit(1)

    target_language = (target_language or "").strip()
    if not target_language:
        print("[错误] --translate-to 不能为空", file=sys.stderr)
        sys.exit(1)

    detected_lang = result.get("language", "unknown")
    translator_source = source_language or "auto"
    if translator_source == "auto" and detected_lang != "unknown":
        translator_source = detected_lang

    try:
        translator = GoogleTranslator(source=translator_source, target=target_language)
    except Exception as err:
        print(f"[错误] 初始化翻译器失败: {err}", file=sys.stderr)
        sys.exit(1)

    segments = result.get("segments", [])
    total = len(segments)
    print(f"[4/4] 翻译中 ({translator_source} -> {target_language}, {total} 段) ...")
    t0 = time.time()

    translated_segments = []
    for idx, seg in enumerate(segments, 1):
        src_text = seg.get("text", "").strip()
        translated_text = src_text
        if src_text:
            chunks = split_text_by_words(src_text, max_chars=4000)
            translated_chunks = [translate_with_retry(translator, chunk) for chunk in chunks]
            translated_text = " ".join(part.strip() for part in translated_chunks if part.strip()).strip()
            if not translated_text:
                translated_text = src_text

        translated_seg = dict(seg)
        translated_seg["text"] = translated_text
        translated_segments.append(translated_seg)

        if idx % 100 == 0 or idx == total:
            print(f"       已翻译 {idx}/{total} 段")

    translated_result = dict(result)
    translated_result["source_language"] = detected_lang
    translated_result["language"] = target_language
    translated_result["segments"] = translated_segments
    translated_result["text"] = " ".join(
        seg["text"].strip() for seg in translated_segments if seg.get("text", "").strip()
    ).strip()

    print(f"       翻译完成 ({time.time() - t0:.1f}s)")
    return translated_result


def write_txt(result: dict, output_path: str) -> None:
    """输出纯文本（无时间戳）。"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip() + "\n")


def write_srt(result: dict, output_path: str) -> None:
    """输出 SRT 字幕格式。"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            start = format_timestamp(seg["start"], style="srt")
            end = format_timestamp(seg["end"], style="srt")
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def write_json(result: dict, output_path: str) -> None:
    """输出 JSON 结构化数据。"""
    data = {
        "language": result.get("language", "unknown"),
        "text": result["text"],
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result["segments"]
        ],
    }
    if result.get("source_language"):
        data["source_language"] = result["source_language"]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_md(result: dict, output_path: str, input_path: str) -> None:
    """输出 Markdown 文档（带时间戳段落）。"""
    segments = result.get("segments", [])
    detected_lang = result.get("language", "unknown")
    source_lang = result.get("source_language")
    input_name = Path(input_path).name

    # 将短片段合并为较长的段落（目标约 60 秒一段）
    paragraphs = []
    current_texts = []
    current_start = 0.0
    current_end = 0.0

    for seg in segments:
        if not current_texts:
            current_start = seg["start"]
        current_texts.append(seg["text"].strip())
        current_end = seg["end"]

        if current_end - current_start >= 60:
            paragraphs.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_texts),
            })
            current_texts = []

    if current_texts:
        paragraphs.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_texts),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 转录文档: {input_name}\n\n")
        f.write(f"> 检测语言: {detected_lang}  \n")
        if source_lang:
            f.write(f"> 源语言: {source_lang}  \n")
        f.write(f"> 总段落数: {len(paragraphs)}  \n")
        f.write(f"> 总片段数: {len(segments)}  \n\n")
        f.write("---\n\n")

        for para in paragraphs:
            ts = format_timestamp(para["start"], style="human")
            f.write(f"**[{ts}]**\n\n")
            f.write(para["text"] + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="视频/音频转录工具 —— 基于 OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  %(prog)s video.mp4                           # 默认 large 模型, 自动检测语言, 输出 .md
  %(prog)s video.mp4 --model medium            # 使用 medium 模型 (更快)
  %(prog)s video.mp4 --language zh             # 强制中文
  %(prog)s video.mp4 --translate               # 翻译为英文
  %(prog)s video.mp4 --language en --translate-to zh-CN  # 先英文转录，再翻译到中文
  %(prog)s video.mp4 --format srt              # 输出 SRT 字幕
  %(prog)s video.mp4 --format json             # 输出 JSON
  %(prog)s video.mp4 --output my_result.md     # 指定输出文件名
  %(prog)s video.mp4 --gpu 2                   # 使用 GPU 2
  %(prog)s video.mp4 --gpu -1                  # 使用 CPU
  %(prog)s audio.mp3 --prompt "以下是会议记录"  # 提供初始提示词
  %(prog)s video.mp4 --model-dir /data/whisper  # 自定义模型缓存路径
  WHISPER_CACHE_DIR=/data/whisper %(prog)s video.mp4  # 通过环境变量指定
""",
    )
    parser.add_argument("input", help="输入的视频或音频文件路径")
    parser.add_argument(
        "--model", "-m",
        default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小 (默认: large)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="强制语言代码, 如 zh/en/ja (默认: 自动检测)",
    )
    parser.add_argument(
        "--translate", "-t",
        action="store_true",
        help="翻译为英文 (默认: 原语言转录)",
    )
    parser.add_argument(
        "--translate-to",
        default=None,
        help="二次翻译目标语言代码, 如 zh-CN/en/ja (先转录再翻译, 需 deep-translator)",
    )
    parser.add_argument(
        "--format", "-f",
        default="md",
        choices=["txt", "srt", "md", "json"],
        help="输出格式 (默认: md)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出文件路径 (默认: 与输入同名, 扩展名按格式变化)",
    )
    parser.add_argument(
        "--gpu", "-g",
        type=int,
        default=0,
        help="GPU 编号, -1 表示 CPU (默认: 0)",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="初始提示词, 可引导模型使用特定术语或风格",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="模型下载/缓存目录 (默认: 环境变量 WHISPER_CACHE_DIR 或 ~/.cache/whisper)",
    )

    args = parser.parse_args()

    if args.translate and args.translate_to:
        parser.error("--translate 与 --translate-to 不能同时使用")

    if not os.path.isfile(args.input):
        print(f"[错误] 找不到文件: {args.input}", file=sys.stderr)
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        stem = Path(args.input).stem
        output_path = f"{stem}_transcription.{args.format}"

    print(f"{'=' * 50}")
    print(f"  视频/音频转录工具")
    print(f"{'=' * 50}")
    print(f"  输入:   {args.input}")
    print(f"  输出:   {output_path}")
    print(f"  模型:   {args.model}")
    print(f"  语言:   {args.language or '自动检测'}")
    print(f"  Whisper翻译(到英文): {'是' if args.translate else '否'}")
    print(f"  二次翻译目标: {args.translate_to or '无'}")
    print(f"  格式:   {args.format}")
    print(f"  GPU:    {'CPU' if args.gpu < 0 else args.gpu}")
    resolved_dir = resolve_model_dir(args.model_dir)
    print(f"  模型目录: {resolved_dir or '~/.cache/whisper (默认)'}")
    if args.prompt:
        print(f"  提示词: {args.prompt[:50]}...")
    print(f"{'=' * 50}\n")

    result = transcribe(
        input_path=args.input,
        model_name=args.model,
        language=args.language,
        translate=args.translate,
        gpu=args.gpu,
        initial_prompt=args.prompt,
        model_dir=args.model_dir,
    )

    if args.translate_to:
        if args.language is None:
            print("       提示: 未指定 --language，先自动检测语种再翻译。若源视频主要为英文，建议加 --language en。")
        result = translate_result_segments(
            result=result,
            target_language=args.translate_to,
            source_language=args.language or "auto",
        )

    writers = {
        "txt": write_txt,
        "srt": write_srt,
        "md": write_md,
        "json": write_json,
    }

    if args.format == "md":
        writers["md"](result, output_path, args.input)
    else:
        writers[args.format](result, output_path)

    file_size = os.path.getsize(output_path)
    print(f"\n[完成] 已保存到 {output_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
