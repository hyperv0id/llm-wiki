#!/usr/bin/env python3
"""Lint script for llm-wiki. Covers the .agents/skills/wiki-lint/SKILL.md checklist."""
import os, re, json, datetime, sys
from collections import defaultdict

WIKI = os.path.join(os.path.dirname(__file__), "..", "wiki")
TODAY = datetime.date.today()

# ---------- load all pages ----------
pages = {}  # name (no ext) -> {path, frontmatter, body}
fm_re = re.compile(r"^---\n(.*?)\n---\n", re.S)
for root, dirs, files in os.walk(WIKI):
    for f in files:
        if not f.endswith(".md"):
            continue
        path = os.path.join(root, f)
        name = f[:-3]
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        m = fm_re.match(text)
        fm = {}
        if m:
            # naive YAML scalar parse (good enough for this repo's flat keys)
            for line in m.group(1).split("\n"):
                mm = re.match(r"^(\w+):\s*(.*)$", line)
                if mm:
                    fm[mm.group(1)] = mm.group(2).strip().strip('"').strip("'")
        pages[name] = {"path": os.path.relpath(path, WIKI), "fm": fm,
                       "body": text[m.end():] if m else text}

ALLOWED_TYPES = {"concept", "technique", "entity", "source-summary", "analysis"}
severe, warn, info = [], [], []

def add(bucket, page, msg):
    bucket.append(f"{page} — {msg}")

# ---------- structure checks ----------
for name, p in sorted(pages.items()):
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

# ---------- citation checks ----------
src_use_re = re.compile(r"\[\^src-([a-z0-9-]+)\](?::)?")
src_def_re = re.compile(r"^\[\^src-([a-z0-9-]+)\]:", re.M)
link_in_def_re = re.compile(r"^\[\^src-[a-z0-9-]+\]:\s*\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", re.M)
general_re = re.compile(r"大致上|一般来说|通常来说|总的来说|普遍而言|广义上")

for name, p in sorted(pages.items()):
    body = p["body"]
    used = set(src_use_re.findall(body))
    defined = set(src_def_re.findall(body))
    # inline uses exclude definition lines
    body_wo_defs = src_def_re.sub("", body)
    used_inline = set(src_use_re.findall(body_wo_defs))
    if used_inline - defined:
        add(severe, name, f"内联引用未定义: {sorted(used_inline - defined)[:5]}")
    if defined - used:
        add(warn, name, f"底部定义但正文未引用: {sorted(defined - used)[:5]}")
    # defined source-summary page must exist
    missing_pages = [s for s in defined if f"source-{s}" not in pages]
    # footnote def is [[source-X]] — the slug inside src-X already includes... check mapping below
    for s in sorted(defined):
        # definition line target
        pass
    sc = p["fm"].get("source_count")
    try:
        sc = int(sc)
    except (TypeError, ValueError):
        add(severe, name, f"source_count 缺失或非法: {sc!r}")
        sc = None
    if sc is not None and sc != len(defined):
        add(warn, name, f"source_count={sc} 但实际定义引用 {len(defined)} 个")
    # timeliness
    lu = p["fm"].get("last_updated", "")
    if p["fm"].get("status") == "active" and re.match(r"\d{4}-\d{2}-\d{2}$", lu):
        d = datetime.date.fromisoformat(lu)
        if (TODAY - d).days > 30:
            add(info, name, f"last_updated={lu} 超 30 天未更新且 status: active")
    if sc == 1 and general_re.search(body):
        add(warn, name, "source_count=1 但含概括性陈述（大致上/一般来说等）")
    if p["fm"].get("confidence") == "high" and (sc or 0) < 2:
        add(severe, name, f"confidence: high 但 source_count={sc}")

# footnote definitions point to [[source-slug]]; verify those targets exist
for name, p in sorted(pages.items()):
    for tgt in link_in_def_re.findall(p["body"]):
        if tgt.strip() not in pages:
            add(severe, name, f"脚注定义指向不存在的 source-summary: {tgt.strip()}")

# ---------- link checks ----------
inbound = defaultdict(int)
wl_re = re.compile(r"\[\[([^\]#|^]+)(?:[|#][^\]]*)?\]\]")
for name, p in pages.items():
    for tgt in wl_re.findall(p["body"]):
        tgt = tgt.strip()
        if tgt in pages and tgt != name:
            inbound[tgt] += 1
for name in sorted(pages):
    if inbound[name] == 0:
        add(warn, name, "孤立页面（无任何其他页面链入）")

print(json.dumps({"total_pages": len(pages),
                  "severe": severe, "warn": warn, "info": info}, ensure_ascii=False, indent=1))
