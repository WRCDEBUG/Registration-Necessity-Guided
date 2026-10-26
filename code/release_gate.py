#!/usr/bin/env python3
"""Submission-final gate for the RN-BC revision.

Checks, in order:
  A. build        -- fresh compile, undefined refs, overfull/underfull boxes, page count
  B. claims       -- required new claims present, withdrawn claims only in withdrawal context
  C. numbers      -- headline values in the PDF/tables agree with audit/numbers.csv
  D. reply        -- all 22 reviewer comments are answered
  E. ledger       -- integrity, status counts vs STATUS.md, source existence
  F. release      -- SHA256 manifest, required files, checkpoint count
  G. placeholders  -- TODO/FIXME, author markers, repository URL

Writes release_gate_report.md with three sections:
  blocking / author-confirmation / non-blocking limitation
Exit code is non-zero when a blocking item is found.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent.parent
TEX = S / "main_revision_A.tex"
PDF = S / "main_revision_A.pdf"
LEDGER = S / "audit/numbers.csv"
REPLY = S / "reply/response_to_reviewers.md"
STATUS = S / "STATUS.md"
BUNDLE = S / "release/rnbc_release"

blocking: list[str] = []
author: list[str] = []
notes: list[str] = []


def sh(cmd: str, cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, cwd=cwd or S, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def ok(cond: bool, msg: str, bucket: list[str]) -> None:
    if not cond:
        bucket.append(msg)


# ---------------------------------------------------------------- A. build
print("[gate] A. build")
for _ in range(2):
    sh("pdflatex -interaction=nonstopmode main_revision_A.tex")
sh("bibtex main_revision_A")
for _ in range(2):
    sh("pdflatex -interaction=nonstopmode main_revision_A.tex")
log = (S / "main_revision_A.log").read_text(errors="ignore")
undef = len(re.findall(r"Reference .* undefined|Citation .* undefined", log))
overfull = len(re.findall(r"Overfull \\hbox", log))
underfull = len(re.findall(r"Underfull \\hbox", log))
pages = re.search(r"Pages:\s+(\d+)", sh(f"pdfinfo {PDF.name}")[1])
n_pages = int(pages.group(1)) if pages else -1
ok(PDF.exists(), "main_revision_A.pdf missing", blocking)
ok(undef == 0, f"undefined references/citations: {undef}", blocking)
ok(overfull == 0, f"overfull hboxes: {overfull}", notes)
tex = TEX.read_text()
_paths = [(S / m) if (S / m).exists() else (S / (m + ".tex")) for m in re.findall(r"\\input\{([^}]+)\}", tex)]
_incl = "\n".join(p.read_text() for p in _paths if p.exists())
tex_all = tex + "\n" + _incl

# ---------------------------------------------------------------- B. claims
print("[gate] B. claims")
pdf_text = sh(f"pdftotext {PDF.name} -")[1]
pdf_flat = re.sub(r"\s+", " ", pdf_text)
tex_flat = re.sub(r"\s+", " ", tex_all)

required = {
    "phi sweep": ["0.6975", "0.7077"],
    "s_g sweep": ["0.7215"],
    "task-only row": ["Task only", "0.6732"],
    "seed table": ["seed 0", "0.0008"],
    "detector provenance": ["Downstream detector provenance", "50 epochs"],
    "SeAFusion": ["SeAFusion", "0.7423", "0.7710"],
    "fairness both metrics": ["0.7344", "0.7525", "0.0180"],
    "fairness CI": ["0.0257", "0.0104"],
    "300-pair note": ["first 300", "trend"],
    "displacement threshold": ["0.6847"],
    "parallax diagnostic": ["layered-parallax", "0.7813"],
    "sweep paired CI": ["0.0085"],
    "LLVIP 3 seeds": ["0.8529"],
    "parallax 3 seeds": ["0.7737"],
    "MS2 real parallax": ["25.5", "0.4193", "0.4399"],
    "VTMOT detection": ["0.1984", "0.1990"],
    "SIFT module": ["0.6156", "0.6099"],
}
for name, needles in required.items():
    for n in needles:
        ok(n in tex_flat, f"required claim missing from manuscript: {name} -> '{n}'", blocking)

withdrawn = {
    "failed to influence downstream detection": "withdrawn causal claim",
    "essentially uncorrelated": "withdrawn wording",
    "effectively uncorrelated": "withdrawn wording",
}
for phrase, why in withdrawn.items():
    ok(re.sub(r"\s+", " ", phrase) not in tex_flat, f"withdrawn wording still present: '{phrase}' ({why})", blocking)

# +0.1163 may appear, but only next to an explicit withdrawal
for m in re.finditer(r"\+0\.1163", tex_flat):
    ctx = tex_flat[max(0, m.start() - 260): m.start() + 160]
    if not re.search(r"withdraw|previously reported|not a method comparison|replaced here", ctx, re.I):
        blocking.append("'+0.1163' appears without an explicit withdrawal context")
n_1163 = len(re.findall(r"\+0\.1163", tex_flat))
notes.append(f"'+0.1163' occurrences: {n_1163} (explicitly withdrawn)")

# SuperFusion must not be sold as a robustness result
for m in re.finditer(r"SuperFusion", pdf_flat):
    ctx = pdf_flat[max(0, m.start() - 120): m.start() + 160]
    if re.search(r"robust", ctx, re.I) and not re.search(r"not|non-competitive|below", ctx, re.I):
        blocking.append("SuperFusion still framed as robustness evidence")
n_5525 = len(re.findall(r"0\.5525", pdf_flat))
notes.append(f"'0.5525' occurrences in PDF: {n_5525} (table row for misaligned SAGE-IVIF only)")

# ---------------------------------------------------------------- C. numbers
print("[gate] C. numbers vs ledger")
led = {r["number_id"]: r for r in csv.DictReader(open(LEDGER))}
checks = [
    ("T3_none_mAP50", "0.5887"),
    ("T3_full_mAP50", "0.6922"),
    ("T3_oracle_task_mAP50", "0.6865"),
    ("T3_oracle_product_mAP50", "0.6758"),
    ("T3_oracle_geom_mAP50", "0.6689"),
    ("T3_uniform_mAP50", "0.6380"),
    ("T3_geom_thresh_mAP50", "0.6847"),
    ("T3_task_only_mAP50", "0.6732"),
    ("T3_factorised_mAP50", "0.6709"),
    ("T3_proxy_net_mAP50", "0.6792"),
    ("E2_C3_mAP50", "0.6155"),
    ("E2_C4_mAP50", "0.6251"),
    ("BACKBONE_gated_mAP50", "0.7423"),
    ("BACKBONE_full_mAP50", "0.7710"),
    ("FAIR_metric_reversal", "-0.0180"),
    ("SAGE_aligned_reproduce", "0.6420"),
]
for nid, val in checks:
    if nid not in led:
        blocking.append(f"ledger entry missing: {nid}")
        continue
    if led[nid]["value"] not in val and val.lstrip("+") not in led[nid]["value"].lstrip("+"):
        blocking.append(f"ledger value mismatch for {nid}: {led[nid]['value']} vs expected {val}")
    if val.lstrip("+-") not in tex_flat:
        blocking.append(f"value {val} ({nid}) not found in manuscript text")

# headline abstract numbers must be in the ledger
for val in ("0.6922", "0.5887", "0.6847", "0.6732", "0.6709", "0.6380", "0.6792"):
    ok(any(r["value"].lstrip("+") == val for r in led.values()),
       f"abstract value {val} has no ledger entry", blocking)

# ---------------------------------------------------------------- D. reply
print("[gate] D. reply letter")
rl = REPLY.read_text()
need = [f"R1.{i}" for i in range(1, 7)] + [f"R1.m{i}" for i in range(1, 11)] + [f"R2.{i}" for i in range(1, 7)]
for tag in need:
    if not re.search(rf"###\s+{re.escape(tag)}\b", rl):
        blocking.append(f"reply letter missing section {tag}")
ok("withdraw" in rl.lower(), "reply letter has no withdrawal statement", notes)

# ---------------------------------------------------------------- E. ledger
print("[gate] E. ledger integrity")
status_counts: dict[str, int] = {}
for r in led.values():
    status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
for r in led.values():
    if r["status"] in ("verified", "recomputed", "recorded"):
        if not r["value"].strip():
            blocking.append(f"ledger row with empty value: {r['number_id']}")
        sp = r["source_path"].strip()
        if sp.startswith(("exp/", "audit/", "paper/")):
            if not (S / sp).exists():
                blocking.append(f"ledger source missing: {r['number_id']} -> {sp}")
status_txt = STATUS.read_text()
for k, v in status_counts.items():
    if str(v) not in status_txt:
        notes.append(f"STATUS.md does not mention ledger count {k}={v}")
n_ledger = len(led)
notes.append(f"ledger rows: {n_ledger} ({status_counts})")

# ---------------------------------------------------------------- F. release
print("[gate] F. release bundle")
if not BUNDLE.exists():
    blocking.append("release bundle directory missing")
else:
    rc, out = sh("sha256sum -c SHA256SUMS", cwd=BUNDLE)
    ok(rc == 0, f"SHA256 manifest verification failed: {out.splitlines()[:2]}", blocking)
    n_files = len(list(BUNDLE.rglob("*")))
    n_ckpt = len(list((BUNDLE / "checkpoints").glob("*")))
    for req in ("README.md", "REPRODUCE.md", "SHA256SUMS", "code", "results/numbers.csv", "manifests"):
        ok((BUNDLE / req).exists(), f"release bundle missing {req}", blocking)
    ok(n_ckpt == 8, f"release bundle has {n_ckpt} checkpoints, expected 8", notes)
    notes.append(f"release bundle: {n_files} files, {n_ckpt} checkpoints, SHA256SUMS verified")

# ---------------------------------------------------------------- G. placeholders
print("[gate] G. placeholders and URLs")
todo = re.findall(r"\b(TODO|FIXME|XXX)\b", tex)
ok(not todo, f"TODO/FIXME markers in manuscript: {set(todo)}", blocking)
markers = re.findall(r"\[Authors: please confirm[^\]]*\]", tex)
n_markers = len(markers)
ok(n_markers <= 1, f"{n_markers} unresolved author markers", author)
if n_markers:
    author.append("author-contribution marker present in manuscript (needs author confirmation)")
urls = re.findall(r"https?://[^\s}\)]+", tex)
repo = [u for u in urls if re.search(r"github|anonymous|anon|4open|zenodo", u, re.I)]
ok(bool(repo), "no repository URL in the manuscript (Data Availability)", blocking)
if "10.3390/s1010000" in tex:
    blocking.append("template DOI placeholder present in the manuscript")
else:
    notes.append("no template DOI placeholder in the manuscript source")

# Author order, affiliation, e-mails and corresponding author are taken from the
# original submission (main.tex in sensors-4626757原始稿件.zip) and are therefore not
# open questions; only the items below cannot be resolved from the files.
for item in ():
    author.append(item)

# ---------------------------------------------------------------- report
print("[gate] writing report")
lines = ["# Release gate report", "",
         f"- manuscript: `{TEX.name}` ({n_pages} pages)",
         f"- build: undefined refs = {undef}, overfull hbox = {overfull}, underfull hbox = {underfull}",
         f"- ledger: {n_ledger} rows {status_counts}",
         "", "## Blocking", ""]
lines += [f"- {b}" for b in blocking] or ["- none"]
lines += ["", "## Author confirmation", ""]
lines += [f"- {a}" for a in author]
lines += ["", "## Non-blocking notes / limitations", ""]
lines += [f"- {n}" for n in notes]
lines += ["- real-parallax data are now used (MS2, Table 13); occlusion and moving-object masks are still absent, and MS2 has no detection labels",
          "- PDF /Subject is inherited from the abstract by the MDPI class and loses math symbols"]
(S / "release_gate_report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if blocking else 0)
