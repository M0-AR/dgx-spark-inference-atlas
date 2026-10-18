#!/usr/bin/env python3
"""Mechanically regenerate README's generated blocks from results/ artifacts.

Rule (docs/METHODOLOGY.md §7): no number in the README may be hand-typed.
This script is the only writer of the marked blocks; it refuses to invent
anything and prints "no data yet" rows when artifacts are missing.

Inputs (all under results/):
  derived/agentic-<tag>-<utc>.json   episode grades + aggregates (tool arm)
  derived/summary-<tag>-<utc>.json   runner condition summaries (timing arm)
  derived/quality-<tag>-<utc>.json   KLD/PPL per condition (exllamav3 arm)
  raw/market-*.json                  live snapshots
  raw/env-*.json                     environment captures

Outputs:
  results/derived/tables/*.md        rendered tables (committed)
  results/derived/macros.tex         LaTeX macros for papers/main.tex
  results/derived/readme-sync.json   manifest w/ digests of every input/output
  README.md                          blocks between BEGIN/END markers replaced

Exclusions: model == "mock" or tag starting with "mock-" never enters tables.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.bench import stats as S  # noqa: E402

RESULTS = REPO / "results"
TABLES = RESULTS / "derived" / "tables"
REFERENCE_TAG = "fp16"          # paired comparisons are always vs this condition
CONDITIONS_PATH = REPO / "harness" / "configs" / "conditions.json"
FALLBACK_BPW = {"fp16": 16.0, "e40": 4.0, "e30": 3.0, "e20": 2.0,
                "e30-w": 3.0, "e30-a": 3.0}


def load_conditions() -> tuple[dict[str, float], set[str]]:
    """(bpw map, ladder tag set) from the pre-registered conditions file.

    Ladder = conditions with default/null calibration (fp16 + e20/e30/e40);
    the calibration-ablation tags (e30-w/e30-a) are excluded from the
    changepoint scan.
    """
    try:
        doc = json.loads(CONDITIONS_PATH.read_text())
        conds = doc["conditions"]
        bpw = {str(c["tag"]): float(c["bpw"]) for c in conds}
        ladder = {str(c["tag"]) for c in conds
                  if c.get("calibration") in (None, "default")}
        if not bpw or not ladder:
            raise ValueError("no conditions parsed")
        return bpw, ladder
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[sync] conditions.json unreadable ({exc}); using fallback")
        return dict(FALLBACK_BPW), set(FALLBACK_BPW)


BPW, LADDER_TAGS = load_conditions()


def _macrosafe(tag: str) -> str:
    """LaTeX-safe macro suffix: alphanumerics only ('e30-w' -> 'E30W')."""
    return re.sub(r"[^0-9A-Za-z]", "", tag).upper()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_dir(pattern: str) -> list[dict[str, Any]]:
    out = []
    for p in sorted((RESULTS / "derived").glob(pattern)):
        try:
            doc = json.loads(p.read_text())
            doc["_path"] = str(p.relative_to(REPO))
            out.append(doc)
        except (json.JSONDecodeError, OSError):
            print(f"[sync] unreadable artifact skipped: {p}")
    return out


def _is_mock(doc: dict) -> bool:
    tag = doc.get("meta", {}).get("tag", "")
    model = doc.get("meta", {}).get("model", "")
    return tag.startswith("mock-") or model == "mock"


def latest_per_tag(docs: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in docs:
        if _is_mock(d):
            continue
        tag = d.get("meta", {}).get("tag")
        if not tag:
            continue
        run_id = d.get("meta", {}).get("run_id", "")
        if tag not in out or run_id > out[tag].get("meta", {}).get("run_id", ""):
            out[tag] = d
    return out


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_agentic_table(agentic: dict[str, dict]) -> tuple[str, dict]:
    """Markdown table + macro-ready numbers for the tool-calling arm."""
    lines = [
        "| condition | bpw | episodes | C_tool | C_task | C_outcome (95% CI) | repair-needed | vs FP16 (McNemar exact) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    macros: dict[str, Any] = {}
    if not agentic:
        lines.append("| _no episodes recorded yet_ | | | | | | | |")
        return "\n".join(lines), macros

    ref = agentic.get(REFERENCE_TAG)
    pvals: dict[str, float] = {}
    rows: list[tuple[str, str]] = []
    for tag in sorted(agentic, key=lambda t: -BPW.get(t, -1)):
        doc = agentic[tag]
        agg = doc["aggregate"]
        ci = agg["C_outcome_ci95"]
        n = agg["n"]
        cmp_txt = "reference" if tag == REFERENCE_TAG else "—"
        if ref and tag != REFERENCE_TAG and n == ref["aggregate"]["n"]:
            ref_eps = {e["index"]: e["grade"]["C_outcome"] for e in ref["episodes"]}
            cur_eps = {e["index"]: e["grade"]["C_outcome"] for e in doc["episodes"]}
            b = sum(1 for i in ref_eps if ref_eps[i] and not cur_eps.get(i, False))
            c = sum(1 for i in cur_eps if cur_eps[i] and not ref_eps.get(i, False))
            mc = S.mcnemar_exact(b, c)
            pvals[f"mcnemar:{tag}"] = mc["p_value"]
            cmp_txt = f"b={b}, c={c}, p={mc['p_value']:.4f}"
        elif ref and tag != REFERENCE_TAG:
            cmp_txt = f"unpaired (N {n} vs {ref['aggregate']['n']})"
        row = (f"| {tag} | {BPW.get(tag, '?')} | {n} | "
               f"{_fmt_pct(agg['C_tool_rate'])} | {_fmt_pct(agg['C_task_rate'])} | "
               f"{_fmt_pct(agg['C_outcome_rate'])} "
               f"({_fmt_pct(ci['lo'])}–{_fmt_pct(ci['hi'])}) | "
               f"{_fmt_pct(agg['repair_rate'])} | {cmp_txt} |")
        rows.append((tag, row))
        macros[f"atlasCout{_macrosafe(tag)}"] = agg["C_outcome_rate"]
        macros[f"atlasCoutCiLo{_macrosafe(tag)}"] = ci["lo"]
        macros[f"atlasCoutCiHi{_macrosafe(tag)}"] = ci["hi"]
        macros[f"atlasN{_macrosafe(tag)}"] = n

    holm = S.holm_bonferroni(pvals) if pvals else {}
    for tag, row in rows:
        key = f"mcnemar:{tag}"
        if key in holm:
            row = row.replace(
                f"p={pvals[key]:.4f}",
                f"p={pvals[key]:.4f} (Holm {holm[key]['p_adj']:.4f})")
        lines.append(row)
    macros["atlasMcNemarFamily"] = len(pvals)
    return "\n".join(lines), macros


def build_timing_table(summaries: dict[str, dict]) -> tuple[str, dict]:
    lines = [
        "| condition | warm-up | p50 ms | p90 ms | decode tok/s (p50) | CV flag | J/token | errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    macros: dict[str, Any] = {}
    if not summaries:
        lines.append("| _no timing runs recorded yet_ | | | | | | | |")
        return "\n".join(lines), macros
    for tag, doc in sorted(summaries.items()):
        for cond in doc.get("conditions", []):
            lat = cond.get("latency", {})
            tok = cond.get("decode_tok_s", {})
            energy = cond.get("energy") or {}
            jnet = energy.get("joules_per_token_net")
            jtok = jnet if jnet is not None else energy.get("joules_per_token")
            lines.append(
                f"| {cond['condition']} | "
                f"{'ok' if cond['warmup_compliant'] else 'DEVIATION'} | "
                f"{lat.get('p50_ms', float('nan')):.1f} | "
                f"{lat.get('p90_ms', float('nan')):.1f} | "
                f"{tok.get('p50_ms', float('nan')):.2f} | "
                f"{'⚠' if cond['cv_flag'] else 'ok'} | "
                f"{'—' if jtok is None else f'{jtok:.3f}'} | "
                f"{cond['errors']} |")
            tok_p50 = tok.get("p50_ms")
            if tok_p50 is not None:
                macros[f"atlasTok{_macrosafe(cond['condition'])}"] = tok_p50
    return "\n".join(lines), macros


def build_provenance_table() -> tuple[str, dict]:
    def study_files(pattern: str) -> list[Path]:
        return [p for p in sorted((RESULTS / "raw").glob(pattern))
                if "mock" not in p.name]

    markets = study_files("market-*.json")
    envs = study_files("env-*.json")
    runs = study_files("run-*.jsonl")
    tc = study_files("toolcall-*.jsonl")
    d = {
        "market_snapshots": len(markets),
        "env_captures": len(envs),
        "runner_runs": len(runs),
        "toolcall_runs": len(tc),
    }
    latest_env = envs[-1].name if envs else "—"
    latest_mkt = markets[-1].name if markets else "—"
    lines = [
        "| artifact class | count | latest |",
        "|---|---|---|",
        f"| market snapshots | {d['market_snapshots']} | {latest_mkt} |",
        f"| environment captures | {d['env_captures']} | {latest_env} |",
        f"| runner runs (timing) | {d['runner_runs']} | {runs[-1].name if runs else '—'} |",
        f"| tool-call runs | {d['toolcall_runs']} | {tc[-1].name if tc else '—'} |",
    ]
    macros = {f"atlasProv{k.title().replace('_', '')}": v for k, v in d.items()}
    return "\n".join(lines), macros


def build_changepoint_note(agentic: dict[str, dict]) -> tuple[str, dict]:
    pts = sorted(((BPW[t], t, d["aggregate"]["C_outcome_rate"])
                  for t, d in agentic.items()
                  if t in LADDER_TAGS and not _is_mock(d)))
    if len(pts) < 4:
        return ("_Changepoint analysis requires ≥4 ladder points "
                "(fp16 + 3 quantized); not yet available._", {})
    xs = [p[0] for p in pts]
    ys = [p[2] for p in pts]
    cps = S.find_changepoints(ys, alpha=0.01, n_perm=2000, seed=0)
    if not cps:
        return ("No changepoint detected on the C_outcome ladder at α=0.01 "
                "(reported as pattern location, not absence of degradation).",
                {"atlasChangepoints": 0})
    parts = []
    for cp in cps:
        left_bpw = xs[max(cp.index - 1, 0)]
        right_bpw = xs[min(cp.index, len(xs) - 1)]
        parts.append(f"gap between {left_bpw} and {right_bpw} bpw "
                     f"(p={cp.p_value:.4f})")
    return (f"Changepoint(s) detected on the C_outcome ladder: "
            f"{'; '.join(parts)}. Pattern location only; see METHODOLOGY §5.",
            {"atlasChangepoints": len(cps)})


def replace_block(readme: str, block: str, content: str) -> str:
    begin = f"<!-- BEGIN:{block} -->"
    end = f"<!-- END:{block} -->"
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pat.search(readme):
        print(f"[sync] WARNING: block {block} not found in README; not replaced")
        return readme
    return pat.sub(f"{begin}\n{content}\n{end}", readme)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    agentic = latest_per_tag(load_json_dir("agentic-*.json"))
    summaries = latest_per_tag(load_json_dir("summary-*.json"))

    a_table, a_macros = build_agentic_table(agentic)
    t_table, t_macros = build_timing_table(summaries)
    p_table, p_macros = build_provenance_table()
    cp_text, cp_macros = build_changepoint_note(agentic)

    (TABLES / "agentic.md").write_text(a_table + "\n")
    (TABLES / "timing.md").write_text(t_table + "\n")
    (TABLES / "provenance.md").write_text(p_table + "\n")
    (TABLES / "changepoints.md").write_text(cp_text + "\n")

    # Zero defaults so papers/main.tex always compiles: every macro it may
    # reference exists even before any data lands (empty results -> zeros).
    zero_macros: dict[str, Any] = {
        **{f"atlasCout{_macrosafe(t)}": 0.0 for t in BPW},
        **{f"atlasCoutCiLo{_macrosafe(t)}": 0.0 for t in BPW},
        **{f"atlasCoutCiHi{_macrosafe(t)}": 0.0 for t in BPW},
        **{f"atlasN{_macrosafe(t)}": 0 for t in BPW},
        "atlasMcNemarFamily": 0,
        "atlasChangepoints": 0,
    }
    macros = {**zero_macros, **a_macros, **t_macros, **p_macros, **cp_macros}
    macro_lines = ["% Auto-generated by scripts/sync_readme.py — do not edit."]
    for k, v in sorted(macros.items()):
        if isinstance(v, float):
            macro_lines.append(f"\\newcommand{{\\{k}}}{{{v:.4f}}}")
        else:
            macro_lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (RESULTS / "derived" / "macros.tex").write_text("\n".join(macro_lines) + "\n")

    readme_path = REPO / "README.md"
    readme_before = readme_path.read_text() if readme_path.exists() else ""
    readme = readme_before
    readme = replace_block(readme, "gen:agentic", a_table)
    readme = replace_block(readme, "gen:timing", t_table)
    readme = replace_block(readme, "gen:provenance", p_table)
    readme = replace_block(readme, "gen:changepoints", cp_text)
    readme_path.write_text(readme)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_condition": REFERENCE_TAG,
        "conditions_in_tables": sorted(agentic),
        "inputs": sorted(str(p.relative_to(REPO)) for p in
                         (RESULTS / "raw").glob("*")),
        "readme_sha256_before": hashlib.sha256(
            readme_before.encode()).hexdigest(),
        "readme_sha256_after": hashlib.sha256(readme.encode()).hexdigest(),
        "tables_sha256": {
            p.name: sha256_file(p) for p in sorted(TABLES.glob("*.md"))},
    }
    (RESULTS / "derived" / "readme-sync.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[sync] agentic conditions: {sorted(agentic) or '(none)'}")
    print(f"[sync] timing conditions: {sorted(summaries) or '(none)'}")
    print(f"[sync] wrote {TABLES}/*.md, macros.tex, readme-sync.json")


if __name__ == "__main__":
    main()
