#!/usr/bin/env python3
"""
ljg-present 渲染器：orgmode -> JSON slides -> HTML
按 ljg-present/SKILL.md 的映射规则 1:1 渲染 outline。
"""

import json
import re
import sys
from pathlib import Path

# ── org 解析 ──────────────────────────────────────────────────────────────────

def parse_inline_emphasis(text):
    """
    解析行内强调，把 *xxx* ~xxx~ =xxx= 转成 chunks[{t, hl}]。
    保留「」内的内容但不强制 hl。
    """
    chunks = []
    # 同时匹配 *emphasis*, ~code~, =verbatim=
    pattern = re.compile(r'\*([^*\n]+)\*|~([^~\n]+)~|=([^=\n]+)=')
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            chunks.append({"t": text[pos:m.start()]})
        emph = m.group(1) or m.group(2) or m.group(3)
        chunks.append({"t": emph, "hl": True})
        pos = m.end()
    if pos < len(text):
        chunks.append({"t": text[pos:]})
    if not chunks:
        chunks.append({"t": text})
    return chunks


def cjk_len(s):
    """CJK 字符按 1.8 计权，与模板 JS 的 lineCharLen 对齐。"""
    cjk = re.compile(r'[　-〿㐀-䶿一-鿿豈-﫿＀-￯]')
    return sum(1.8 if cjk.match(ch) else 1.0 for ch in s)


def split_paragraph_lines(text, max_chars=40):
    """
    段落按换行拆。单行过长时按句号拆。
    保持原行结构——outline 作者已经用分行表达了节奏。
    """
    raw_lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(raw_lines) <= 1:
        # 单行且太长时按中文句号/问号拆
        if cjk_len(text.strip()) > max_chars:
            sentences = re.split(r'(?<=[。？！])', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        return [text.strip()]
    return raw_lines


def parse_org(content):
    """
    解析 org 内容，返回 (title, subtitle, theme, slides)。
    slides 是 JSON 数组，每个元素符合 ljg-present schema。
    """
    lines = content.split('\n')
    title = ""
    subtitle_parts = []
    theme = "black"  # 默认
    slides = []

    i = 0
    n = len(lines)

    # ── 解析文件级元数据 ──
    while i < n:
        line = lines[i]
        if line.startswith('#+title:'):
            title = line[len('#+title:'):].strip()
        elif line.startswith('#+date:'):
            date = line[len('#+date:'):].strip()
            # 去掉 [ ] 包裹
            date = re.sub(r'[\[\]]', '', date).strip()
            subtitle_parts.append(date)
        elif line.startswith('#+author:'):
            author = line[len('#+author:'):].strip()
            subtitle_parts.insert(0, author)
        elif line.startswith('#+filetags:'):
            tags = line[len('#+filetags:'):].strip()
            # 推断 theme
            if any(t in tags for t in [':share:', ':manifesto:', ':keynote:']):
                theme = "red"
            elif any(t in tags for t in [':critique:', ':warn:', ':rant:']):
                theme = "yellow"
            # :talk: :essay: :think: :learn: :note: :progress: -> 保持默认 black
        elif line.strip() == '':
            pass
        else:
            break
        i += 1

    subtitle = ' · '.join(p for p in subtitle_parts if p)

    # ── 解析正文 ──
    while i < n:
        line = lines[i]

        # 空行跳过
        if line.strip() == '':
            i += 1
            continue

        # 一级标题 → emphasis 封面页
        if line.startswith('* '):
            heading = line[2:].strip()
            slides.append({
                "emphasis": True,
                "lines": [{"indent": 0, "chunks": parse_inline_emphasis(heading)}]
            })
            i += 1
            continue

        # 二级标题 → theme 独占页
        if line.startswith('** '):
            heading = line[3:].strip()
            slides.append({
                "lines": [{"indent": 0, "chunks": parse_inline_emphasis(heading)}]
            })
            i += 1
            continue

        # 三级标题 → theme 独占页（字号自动降档）
        if line.startswith('*** '):
            heading = line[4:].strip()
            slides.append({
                "lines": [{"indent": 0, "chunks": parse_inline_emphasis(heading)}]
            })
            i += 1
            continue

        # 分隔符 → emphasis 休止页
        if line.strip() == '-----':
            slides.append({
                "emphasis": True,
                "lines": [{"indent": 0, "chunks": [{"t": ""}]}]
            })
            i += 1
            continue

        # example 块 → pre 页
        if line.strip().startswith('#+begin_example'):
            i += 1
            pre_lines = []
            while i < n and not lines[i].strip().startswith('#+end_example'):
                pre_lines.append(lines[i])
                i += 1
            i += 1  # 跳过 #+end_example
            pre_text = '\n'.join(pre_lines)
            pre_name = ""
            slides.append({
                "preTitle": pre_name,
                "pre": pre_text
            })
            continue

        # 列表项
        if re.match(r'^\s*[-+]\s', line) or re.match(r'^\s*\d+\.\s', line):
            list_items = []
            while i < n and (re.match(r'^\s*[-+]\s', lines[i]) or re.match(r'^\s*\d+\.\s', lines[i]) or
                             (lines[i].strip() == '' and i + 1 < n and
                              (re.match(r'^\s*[-+]\s', lines[i+1]) or re.match(r'^\s*\d+\.\s', lines[i+1])))):
                if lines[i].strip() == '':
                    i += 1
                    continue
                # 计算缩进级别
                indent_match = re.match(r'^(\s*)', lines[i])
                indent_spaces = len(indent_match.group(1)) if indent_match else 0
                indent_level = min(indent_spaces // 2, 2)
                # 去掉列表标记
                item_text = re.sub(r'^\s*[-+]\s+', '', lines[i])
                item_text = re.sub(r'^\s*\d+\.\s+', '', item_text)
                list_items.append((indent_level, item_text))
                i += 1

            # 分页：≤4 项一页
            page_size = 4
            for page_start in range(0, len(list_items), page_size):
                page_items = list_items[page_start:page_start + page_size]
                slide_lines = []
                for indent, item_text in page_items:
                    slide_lines.append({
                        "indent": indent,
                        "chunks": parse_inline_emphasis(item_text)
                    })
                slides.append({"lines": slide_lines})
            continue

        # 段落 → theme 页
        para_lines = []
        while i < n and lines[i].strip() != '' and not re.match(r'^\*{1,}\s', lines[i]) \
              and not lines[i].strip().startswith('#+begin_example') \
              and not re.match(r'^\s*[-+]\s', lines[i]) \
              and not re.match(r'^\s*\d+\.\s', lines[i]) \
              and lines[i].strip() != '-----':
            para_lines.append(lines[i])
            i += 1

        if para_lines:
            para_text = '\n'.join(para_lines)
            split_lines = split_paragraph_lines(para_text)
            total_len = sum(cjk_len(l) for l in split_lines)
            # 段落尽量完整放一页；只在超过 6 行或总长度 > 120 时才拆
            if len(split_lines) <= 6 and total_len <= 120:
                slide_lines = [{"indent": 0, "chunks": parse_inline_emphasis(l)} for l in split_lines]
                slides.append({"lines": slide_lines})
            else:
                # 按行分组，每页 4 行
                for page_start in range(0, len(split_lines), 4):
                    page_lines = split_lines[page_start:page_start + 4]
                    slide_lines = [{"indent": 0, "chunks": parse_inline_emphasis(l)} for l in page_lines]
                    if page_start + 4 < len(split_lines):
                        last_chunk = slide_lines[-1]["chunks"][-1]
                        last_chunk["t"] = last_chunk["t"] + " ⋯"
                    slides.append({"lines": slide_lines})
            continue

        i += 1

    return title, subtitle, theme, slides


# ── HTML 装配 ─────────────────────────────────────────────────────────────────

TEMPLATE_PATH = Path("/home/jcheng/Documents/llm-wiki/.agents/skills/ljg-present/assets/slogan_template.html")

def assemble_html(title, subtitle, theme, slides):
    template = TEMPLATE_PATH.read_text(encoding='utf-8')
    # JSON.stringify 等价
    slides_json = json.dumps(slides, ensure_ascii=False, indent=None, separators=(',', ':'))
    html = template.replace('{{TITLE}}', title) \
                   .replace('{{SUBTITLE}}', subtitle) \
                   .replace('{{THEME}}', theme) \
                   .replace('{{SLIDES_JSON}}', slides_json)
    return html


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    org_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/jcheng/Documents/llm-wiki/notes/20260705-progress-report__presentation.org"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/jcheng/Downloads")
    out_dir.mkdir(parents=True, exist_ok=True)

    content = Path(org_path).read_text(encoding='utf-8')
    title, subtitle, theme, slides = parse_org(content)

    # 输出文件名：取 title，去标点，≤20 字
    safe_name = re.sub(r'[^\w\u4e00-\u9fff-]', '-', title).strip('-')[:30]
    out_path = out_dir / f"{safe_name}.html"

    html = assemble_html(title, subtitle, theme, slides)
    out_path.write_text(html, encoding='utf-8')

    print(f"✓ 生成完成")
    print(f"  源文件：{org_path}")
    print(f"  输出：{out_path}")
    print(f"  主题：{theme}")
    print(f"  页数：{len(slides)}")
    print(f"  翻页键：→ ← Space F Home End")


if __name__ == '__main__':
    main()
