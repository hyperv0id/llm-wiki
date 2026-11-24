#!/usr/bin/env python3
"""Lint & quality gate script for llm-wiki.

Covers:
1. Structural / citation / link checks (.agents/skills/wiki-lint/SKILL.md)
2. Quality & anti-gaming checks (--quality / --only-quality):
   - Evidence section ratio (> 20%)
   - Oversized experimental score tables (> 5 data rows on single-paper pages)
   - Opening-paragraph readability & mid-sentence source coordinates
   - Micro-decimal arithmetic / mean verification / typo audit dumps
   - Concept/Analysis aggregate pages (>= 4 sources) missing a 2D matrix table
   - Banned phrasing & emoji checks
"""
import argparse
import datetime
import json
import os
import re
import sys
from collections import defaultdict

WIKI = os.path.join(os.path.dirname(__file__), "..", "wiki")
TODAY = datetime.date.today()

ALLOWED_TYPES = {"concept", "technique", "entity", "source-summary", "analysis"}
META_PAGES = {"index", "log", "overview"}

# ---------- regexes ----------
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
SRC_USE_RE = re.compile(r"\[\^src-([a-z0-9-]+)\](?::)?")
SRC_DEF_RE = re.compile(r"^\[\^src-([a-z0-9-]+)\]:", re.M)
LINK_IN_DEF_RE = re.compile(r"^\[\^src-[a-z0-9-]+\]:\s*\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", re.M)
GENERAL_RE = re.compile(r"大致上|一般来说|通常来说|总的来说|普遍而言|广义上")
WL_RE = re.compile(r"\[\[([^\]#|^]+)(?:[|#][^\]]*)?\]\]")

# Quality check regexes
COORD_RE = re.compile(
    r"[（(]\s*(?:§\s*\d|Section\s+\d|Sec\.\s*\d|式\s*[（(]?\d|Eq\.?\s*\d|"
    r"Algorithm\s+\d|Alg\.?\s*\d|Table\s+\d|Fig(?:ure)?\.?\s*\d|Appendix\s+[A-Z0-9])"
    r"[^）)\n]{0,45}[）)]"
)
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)")
HEAVY_MATH_HINT_RE = re.compile(r"[_^\\]")
MICRO_DELTA_RE = re.compile(r"(?:[+−\-]\s*0?\.\d{1,3}|±\s*0?\.\d{1,3})")
NITPICK_RE = re.compile(r"核算.{0,8}均值|验算.{0,8}均值|印刷色彩|颜色丢失|印刷掉色|转换后颜色")
BANNED_WORDS_RE = re.compile(r"打赢|全涨|硬伤|账本|兜底|崩坏|决定成败|值得一篇论文|反超|屈居")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]")
TABLE_SEP_RE = re.compile(r"^\|\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$", re.M)
NUM_CELL_RE = re.compile(r"\d+(?:\.\d+)?")
QUOTED_RE = re.compile(r"「[^」]*」|“[^”]*”|\"[^\"]*\"")


def load_pages():
    pages = {}
    for root, _, files in os.walk(WIKI):
        for f in files:
            if not f.endswith(".md"):
                continue
            path = os.path.join(root, f)
            name = f[:-3]
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            m = FM_RE.match(text)
            fm = {}
            if m:
                for line in m.group(1).split("\n"):
                    mm = re.match(r"^(\w+):\s*(.*)$", line)
                    if mm:
                        fm[mm.group(1)] = mm.group(2).strip().strip('"').strip("'")
            pages[name] = {
                "path": os.path.relpath(path, WIKI),
                "fm": fm,
                "body": text[m.end():] if m else text,
            }
    return pages


def resolve_targets(raw_targets, pages):
    if not raw_targets:
        return sorted(pages.keys())
    resolved = []
    for t in raw_targets:
        base = os.path.basename(t)
        if base.endswith(".md"):
            base = base[:-3]
        if base in pages:
            resolved.append(base)
        else:
            print(f"[lint] warning: target not found in wiki/: {t}", file=sys.stderr)
    return sorted(set(resolved))


def strip_footnotes(body):
    return re.sub(r"^\[\^src-[a-z0-9-]+\]:.*$", "", body, flags=re.M).strip()


def split_h2_sections(body):
    """Return list of (heading_title_or_None, section_text)."""
    lines = body.splitlines()
    sections = []
    cur_title = None
    cur_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("## "):
            sections.append((cur_title, "\n".join(cur_lines)))
            cur_title = line[3:].strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    sections.append((cur_title, "\n".join(cur_lines)))
    return sections


def extract_prose_paragraphs(text):
    """Extract prose paragraphs excluding code fences, display math, tables, lists, blockquotes, headings."""
    # Remove fenced code blocks and display math blocks
    cleaned = re.sub(r"```[\s\S]*?```", "\n\n", text)
    cleaned = re.sub(r"\$\$[\s\S]*?\$\$", "\n\n", cleaned)
    paras = []
    buf = []
    for line in cleaned.splitlines():
        s = line.strip()
        if (
            not s
            or s.startswith("#")
            or s.startswith("|")
            or s.startswith(">")
            or re.match(r"^([-*+]|\d+\.)\s+", s)
            or s.startswith("[^src-")
        ):
            if buf:
                paras.append(" ".join(buf))
                buf = []
        else:
            buf.append(s)
    if buf:
        paras.append(" ".join(buf))
    return paras


def extract_tables(text):
    """Return list of (header_line, data_rows) for markdown tables."""
    cleaned = re.sub(r"```[\s\S]*?```", "", text)
    lines = cleaned.splitlines()
    tables = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1].strip()):
            header = lines[i].strip()
            i += 2
            data_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                data_rows.append(lines[i].strip())
                i += 1
            tables.append((header, data_rows))
        else:
            i += 1
    return tables


def is_numeric_score_table(data_rows):
    if not data_rows:
        return False
    total_cells = 0
    numeric_cells = 0
    for row in data_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        for c in cells[1:]:  # skip row label column
            if not c:
                continue
            total_cells += 1
            if NUM_CELL_RE.search(c):
                numeric_cells += 1
    return total_cells > 0 and (numeric_cells / total_cells) >= 0.45


def run_quality_checks(name, p):
    issues = []
    if name in META_PAGES:
        return issues

    fm = p["fm"]
    ptype = fm.get("type", "")
    body = strip_footnotes(p["body"])
    total_chars = len(re.sub(r"\s+", "", body))
    if total_chars < 200:
        return issues

    sections = split_h2_sections(body)

    # Q1: Evidence section proportion > 20%
    if ptype in {"technique", "entity", "concept", "analysis"} and total_chars >= 600:
        ev_chars = 0
        for title, content in sections:
            if title and re.match(r"^(?:实验)?证据|^实验结果|^实验(?:与分析)?$|^Experiments?$", title, re.I):
                ev_chars += len(re.sub(r"\s+", "", content))
        ratio = ev_chars / total_chars
        if ratio > 0.20:
            issues.append(f"实验章节占比 {ratio * 100:.0f}% (>20%)，疑似抄表或展开过多")

    # Q2: Oversized experimental score tables (> 5 data rows) on single-paper pages
    if ptype in {"technique", "entity", "source-summary"}:
        for title, content in sections:
            in_ev = bool(title and re.match(r"^(?:实验)?证据|^实验", title))
            for _, data_rows in extract_tables(content):
                if len(data_rows) > 5 and (in_ev or is_numeric_score_table(data_rows)):
                    loc = f"（{title}）" if title else ""
                    issues.append(f"单篇页含大实验表{loc}：{len(data_rows)} 行 (>5 行)，禁止整表照搬")

    # Q3: Opening-paragraph readability & mid-sentence source coordinates
    first_page_para_checked = False
    mid_sentence_coords = []
    for sec_idx, (title, content) in enumerate(sections):
        paras = extract_prose_paragraphs(content)
        if not paras:
            continue
        # Check opening paragraph of page and each H2 section
        open_para = paras[0]
        cm = COORD_RE.search(open_para)
        if cm:
            loc = f"「{title}」首段" if title else "全文首段"
            issues.append(f"{loc}含源码坐标 {cm.group(0)}，首段应保持纯人话大义")
        if sec_idx == 0 and not first_page_para_checked:
            first_page_para_checked = True
            heavy_formulas = [
                m.group(0) for m in INLINE_MATH_RE.finditer(open_para)
                if HEAVY_MATH_HINT_RE.search(m.group(1))
            ]
            if len(heavy_formulas) >= 4:
                issues.append(f"全文首段堆砌 {len(heavy_formulas)} 处带下标/算子公式碎片，应先人话大义")

        for para in paras:
            for m in COORD_RE.finditer(para):
                tail = para[m.end():].lstrip()
                if tail and not re.match(r"^(?:\[\^src-|[。！？；;.\n])", tail):
                    mid_sentence_coords.append(m.group(0))

    if len(mid_sentence_coords) >= 3:
        issues.append(f"正文存在 {len(mid_sentence_coords)} 处句中源码坐标打断阅读流（如 {mid_sentence_coords[0]}）")

    # Q4: Micro-decimal arithmetic / mean verification / typo audit dumps
    all_prose = []
    for title, content in sections:
        if title and re.match(r"^(?:待查|勘误|口径备注)", title):
            items = re.findall(r"^(?:\d+\.|[-*])\s+", content, re.M)
            if len(items) >= 4:
                issues.append(f"正文单列长篇「{title}」清单（{len(items)} 条），非关键瑕疵禁入正文")
        for para in extract_prose_paragraphs(content):
            all_prose.append(para)
            # Strip inline math so negative exponents like 10^{-3} aren't miscounted
            para_no_math = INLINE_MATH_RE.sub("", para)
            deltas = MICRO_DELTA_RE.findall(para_no_math)
            if len(deltas) >= 4:
                issues.append(f"单段出现 {len(deltas)} 处小数差值，疑似逐行加减法磨洋工")
                break
    prose_joined = "\n".join(all_prose)
    nm = NITPICK_RE.search(prose_joined)
    if nm:
        issues.append(f"正文含微观验算或排版转换瑕疵描述（命中「{nm.group(0)}」）")

    # Q5: Concept / Analysis aggregate page (>= 4 sources) missing 2D matrix table or using bullet-blurb lists
    try:
        sc = int(fm.get("source_count", 0))
    except (TypeError, ValueError):
        sc = 0
    if ptype in {"concept", "analysis"} and sc >= 4:
        if not TABLE_SEP_RE.search(body):
            issues.append(f"聚合页关联 {sc} 篇源但缺少二维对照矩阵表，疑似段落报菜名")
        long_paper_bullets = [
            line for line in body.splitlines()
            if re.match(r"^\s*[-*]\s+\*\*.*\[\[.+?\]\]", line) and len(line.strip()) >= 120
        ]
        if len(long_paper_bullets) >= 5:
            issues.append(f"聚合页含 {len(long_paper_bullets)} 条长篇逐论文罗列条目，疑似段落报菜名")

    # Q6: Banned words & emoji
    prose_no_quotes = QUOTED_RE.sub("", prose_joined)
    bm = BANNED_WORDS_RE.findall(prose_no_quotes)
    if bm:
        issues.append(f"命中禁用措辞：{sorted(set(bm))}")
    em = EMOJI_RE.findall(body)
    if em:
        issues.append(f"正文含禁用 emoji（{len(em)} 处）")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Lint and quality check for llm-wiki.")
    parser.add_argument("targets", nargs="*", help="Optional wiki page paths or names to inspect.")
    parser.add_argument("--quality", action="store_true", help="Enable readability & anti-gaming quality checks.")
    parser.add_argument("--only-quality", action="store_true", help="Run only quality checks (skip structural checks).")
    parser.add_argument("--text", action="store_true", help="Print human-readable text output instead of JSON.")
    parser.add_argument("--fail-on-quality", action="store_true", help="Exit 1 if any quality issue is found.")
    args = parser.parse_args()

    run_struct = not args.only_quality
    run_qual = args.quality or args.only_quality

    pages = load_pages()
    target_names = resolve_targets(args.targets, pages)

    severe, warn, info, quality = [], [], [], []

    def add(bucket, page, msg):
        bucket.append(f"{page} — {msg}")

    if run_struct:
        # Inbound links computed across full wiki
        inbound = defaultdict(int)
        for name, p in pages.items():
            for tgt in WL_RE.findall(p["body"]):
                tgt = tgt.strip()
                if tgt in pages and tgt != name:
                    inbound[tgt] += 1

        for name in target_names:
            p = pages[name]
            fm = p["fm"]
            if not fm:
                add(severe, name, "缺少 frontmatter")
                continue
            if fm.get("type") not in ALLOWED_TYPES:
                add(severe, name, f"type 非法: {fm.get('type')!r}")
            status = fm.get("status")
            if status == "superseded" and not fm.get("superseded_by"):
                add(severe, name, "status: superseded 但缺 superseded_by")
            if status == "disputed" and "## 争议" not in p["body"]:
                add(severe, name, "status: disputed 但无 '## 争议' 章节")
            sb = fm.get("superseded_by", "")
            if sb:
                target = re.sub(r"^\[\[|\]\]$|^\[\[|\]\].*$", "", sb).split("|")[0].strip()
                target = re.sub(r"\[\[|\]\]", "", target).strip()
                if target and target not in pages:
                    add(severe, name, f"superseded_by 指向不存在的页面: {target}")

            body = p["body"]
            used = set(SRC_USE_RE.findall(body))
            defined = set(SRC_DEF_RE.findall(body))
            body_wo_defs = SRC_DEF_RE.sub("", body)
            used_inline = set(SRC_USE_RE.findall(body_wo_defs))
            if used_inline - defined:
                add(severe, name, f"内联引用未定义: {sorted(used_inline - defined)[:5]}")
            if defined - used:
                add(warn, name, f"底部定义但正文未引用: {sorted(defined - used)[:5]}")

            sc = fm.get("source_count")
            try:
                sc = int(sc)
            except (TypeError, ValueError):
                add(severe, name, f"source_count 缺失或非法: {sc!r}")
                sc = None
            if sc is not None and sc != len(defined):
                add(warn, name, f"source_count={sc} 但实际定义引用 {len(defined)} 个")

            lu = fm.get("last_updated", "")
            if fm.get("status") == "active" and re.match(r"\d{4}-\d{2}-\d{2}$", lu):
                d = datetime.date.fromisoformat(lu)
                if (TODAY - d).days > 30:
                    add(info, name, f"last_updated={lu} 超 30 天未更新且 status: active")
            if sc == 1 and GENERAL_RE.search(body):
                add(warn, name, "source_count=1 但含概括性陈述（大致上/一般来说等）")
            if fm.get("confidence") == "high" and (sc or 0) < 2:
                add(severe, name, f"confidence: high 但 source_count={sc}")

            for tgt in LINK_IN_DEF_RE.findall(body):
                if tgt.strip() not in pages:
                    add(severe, name, f"脚注定义指向不存在的 source-summary: {tgt.strip()}")

            if inbound[name] == 0:
                add(warn, name, "孤立页面（无任何其他页面链入）")

    if run_qual:
        for name in target_names:
            for msg in run_quality_checks(name, pages[name]):
                add(quality, name, msg)

    if args.text:
        print(f"Checked {len(target_names)} page(s) (total wiki: {len(pages)})")
        if run_struct:
            print(f"  severe: {len(severe)} | warn: {len(warn)} | info: {len(info)}")
            for line in severe + warn:
                print(f"  [struct] {line}")
        if run_qual:
            print(f"  quality: {len(quality)}")
            for line in quality:
                print(f"  [quality] {line}")
    else:
        out = {"checked_pages": len(target_names), "total_pages": len(pages)}
        if run_struct:
            out["severe"] = severe
            out["warn"] = warn
            out["info"] = info
        if run_qual:
            out["quality"] = quality
        print(json.dumps(out, ensure_ascii=False, indent=1))

    if args.fail_on_quality and quality:
        sys.exit(1)


if __name__ == "__main__":
    main()
