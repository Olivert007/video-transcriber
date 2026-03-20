#!/usr/bin/env python3
"""
去除转录文本中的时间戳行。

支持常见格式：
1) Markdown 时间戳：**[00:05:22.18]** / [00:05:22.18]
2) SRT 时间范围：00:00:01,234 --> 00:00:05,678
3) VTT 时间范围：00:00:01.234 --> 00:00:05.678

用法示例：
    python remove_timestamps.py input.md
    python remove_timestamps.py input.md -o output.md
    python remove_timestamps.py input.md --inplace
"""

import argparse
import re
import sys
from pathlib import Path


MD_TS_BOLD_RE = re.compile(
    r"^\s*\*\*\[\d{2}:\d{2}:\d{2}(?:[.,]\d{2,3})?\]\*\*\s*$"
)
MD_TS_PLAIN_RE = re.compile(
    r"^\s*\[\d{2}:\d{2}:\d{2}(?:[.,]\d{2,3})?\]\s*$"
)
SRT_OR_VTT_RANGE_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2}[.,]\d{2,3}\s*-->\s*"
    r"\d{2}:\d{2}:\d{2}[.,]\d{2,3}\s*$"
)


def is_timestamp_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        MD_TS_BOLD_RE.match(stripped)
        or MD_TS_PLAIN_RE.match(stripped)
        or SRT_OR_VTT_RANGE_RE.match(stripped)
    )


def squeeze_blank_lines(lines: list[str]) -> list[str]:
    """将连续空行压缩为单个空行。"""
    out: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        out.append(line)
        prev_blank = is_blank
    return out


def strip_timestamps(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    kept_lines: list[str] = []
    removed_count = 0

    for line in lines:
        if is_timestamp_line(line):
            removed_count += 1
            continue
        kept_lines.append(line)

    cleaned_lines = squeeze_blank_lines(kept_lines)
    return "".join(cleaned_lines), removed_count


def resolve_output_path(input_path: Path, output_arg: str | None, inplace: bool) -> Path:
    if inplace:
        return input_path
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_no_ts{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="移除转录文件中的时间戳行")
    parser.add_argument("input", help="输入文件路径（md/txt/srt/vtt）")
    parser.add_argument("-o", "--output", default=None, help="输出文件路径")
    parser.add_argument("--inplace", action="store_true", help="覆盖原文件")
    args = parser.parse_args()

    if args.output and args.inplace:
        parser.error("--output 与 --inplace 不能同时使用")

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[错误] 找不到文件: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = resolve_output_path(input_path, args.output, args.inplace)
    text = input_path.read_text(encoding="utf-8")
    cleaned_text, removed_count = strip_timestamps(text)
    output_path.write_text(cleaned_text, encoding="utf-8")

    print(f"[完成] 已移除时间戳行: {removed_count}")
    print(f"[完成] 输出文件: {output_path}")


if __name__ == "__main__":
    main()
