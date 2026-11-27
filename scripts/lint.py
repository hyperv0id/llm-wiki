#!/usr/bin/env python3
"""Structural & graph consistency linter for llm-wiki.

Checks deterministic invariants only (see .agents/skills/wiki-lint/SKILL.md):
- Frontmatter schema & status dependencies
- Inline [^src-*] references vs footnote definitions & source_count arithmetic
- One-source-per-page invariant for type: source-summary
- Footnote targets, superseded_by targets, orphan pages, freshness
- Generalization claim with single source, high confidence with < 2 sources

Prose quality, readability, and de-jargonization are intentionally NOT checked
here — those belong to independent LLM review (see .agents/skills/wiki-ingest/SKILL.md).
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

FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
SRC_USE_RE = re.compile(r"\[\^src-([a-z0-9-]+)\](?::)?")
SRC_DEF_RE = re.compile(r"^\[\^src-([a-z0-9-]+)\]:", re.M)
LINK_IN_DEF_RE = re.compile(r"^\[\^src-[a-z0-9-]+\]:\s*\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", re.M)
GENERAL_RE = re.compile(r"大致上|一般来说|通常来说|总的来说|普遍而言|广义上")
WL_RE = re.compile(r"\[\[([^\]#|^]+)(?:[|#][^\]]*)?\]\]")


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
        return sorted(n for n in pages.keys() if n not in META_PAGES)
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


def main():
    parser = argparse.ArgumentParser(
        description="Structural & graph consistency check for llm-wiki."
    )
    parser.add_argument("targets", nargs="*", help="Optional wiki page paths or names to inspect.")
    parser.add_argument("--text", action="store_true", help="Print human-readable text output instead of JSON.")
    parser.add_argument("--fail-on-severe", action="store_true", help="Exit 1 if any severe issue is found.")
    args = parser.parse_args()

    pages = load_pages()
    target_names = resolve_targets(args.targets, pages)

    severe, warn, info = [], [], []

    def add(bucket, page, msg):
        bucket.append(f"{page} — {msg}")

    # Inbound links computed across full wiki
    inbound = defaultdict(int)
    for name, p in pages.items():
        for tgt in WL_RE.findall(p["body"]):
            tgt = tgt.strip()
            if tgt in pages and tgt != name:
                inbound[tgt] += 1

    for name in target_names:
        if name in META_PAGES:
            continue
        p = pages[name]
        fm = p["fm"]
        if not fm:
            add(severe, name, "缺少 frontmatter")
            continue
        ptype = fm.get("type")
        if ptype not in ALLOWED_TYPES:
            add(severe, name, f"type 非法: {ptype!r}")
        status = fm.get("status")
        if status == "superseded" and not fm.get("superseded_by"):
            add(severe, name, "status: superseded 但缺 superseded_by")
        if status == "disputed" and "## 争议" not in p["body"]:
            add(severe, name, "status: disputed 但无 '## 争议' 章节")
        sb = fm.get("superseded_by", "")
        if sb:
            target = re.sub(r"\[\[|\]\]", "", sb).split("|")[0].strip()
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

        # One-source-per-page invariant for source-summary
        if ptype == "source-summary" and status != "superseded":
            expected_slug = name[len("source-"):] if name.startswith("source-") else None
            if sc is not None and sc != 1:
                add(severe, name, f"source-summary 应一源一页（source_count=1），当前为 {sc}")
            foreign_refs = sorted(
                s for s in (used_inline | defined)
                if expected_slug is not None and s != expected_slug
            )
            if foreign_refs:
                add(severe, name, f"source-summary 引用了非本源脚注（疑似跨页污染）: {foreign_refs[:5]}")

        lu = fm.get("last_updated", "")
        if status == "active" and re.match(r"\d{4}-\d{2}-\d{2}$", lu):
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

    if args.text:
        print(f"Checked {len(target_names)} page(s) (total wiki: {len(pages)})")
        print(f"  severe: {len(severe)} | warn: {len(warn)} | info: {len(info)}")
        for line in severe:
            print(f"  [severe] {line}")
        for line in warn:
            print(f"  [warn] {line}")
    else:
        out = {
            "checked_pages": len(target_names),
            "total_pages": len(pages),
            "severe": severe,
            "warn": warn,
            "info": info,
        }
        print(json.dumps(out, ensure_ascii=False, indent=1))

    if args.fail_on_severe and severe:
        sys.exit(1)


if __name__ == "__main__":
    main()
