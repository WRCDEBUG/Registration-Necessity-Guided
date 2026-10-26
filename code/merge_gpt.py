#!/usr/bin/env python3
"""Merge the GPT-edited manuscript with the three items it was based before.

Input : main_revision_A_GPT.tex  (author-provided, language-edited version)
Output: main_revision_A.tex      (submission candidate)

The GPT version predates three additions that close specific reviewer points:
  1. detector provenance paragraph  (R2.6)
  2. phi / s_g sensitivity table + text (R1.m3)
  3. author contributions with full names (the "Z.Z." abbreviation is ambiguous
     between Zhaofa Zhou and Zhili Zhang)
and, once available, the paired bootstrap CI for the equal-information-access
comparison (R1.5 / R2.1). Everything else is taken verbatim from the GPT file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SENSE = HERE.parent
SRC = SENSE / "main_revision_A_GPT.tex"
DST = SENSE / "main_revision_A.tex"
BOOT = SENSE / "exp/fair/fairness_bootstrap.json"

s = SRC.read_text()
assert "\\Title{Registration Necessity Is a Magnitude Decision" in s

# 1) detector provenance (before the detection-protocol subsubsection)
PROV = """\t\\paragraph{Downstream detector provenance.}
\tThe detector used for every reported number is a YOLOv8n fine-tuned for the M3FD
\tlabel space. Its configuration was recovered from the checkpoint metadata and
\tcross-checked against the data configuration used for training: initialisation
\tfrom the official YOLOv8n weights, 50 epochs at $640$\\,px with batch size $16$,
\tAdamW with an initial learning rate of $10^{-3}$, weight decay $5\\times10^{-4}$,
\tmosaic augmentation and \\texttt{close\\_mosaic}$=10$, a fixed seed and
\tdeterministic training. The training inputs are the grayscale fused images of the
\t3151 training pairs produced by this project, and the checkpoint was selected by
\tvalidation mAP on the M3FD infrared test split. Two consequences are disclosed
\trather than hidden: the selection uses the split that is later evaluated, and the
\ttraining inputs are fused outputs of this project, which favours pixel-average-like
\tinputs in the comparison of Section~\\ref{sec:fair}. For the cross-detector check we
\tadditionally use an independently trained YOLOv8n on M3FD pixel-average images,
\tand for LLVIP the documented RT-DETR-L detector; both weights, their hashes and
\ttheir training configurations are released with the anonymised repository.

"""
anchor = "\t\\subsubsection{Detection Protocol}"
if PROV not in s:
    assert anchor in s, "detection protocol anchor missing"
    s = s.replace(anchor, PROV + anchor, 1)

# 2) phi / s_g sensitivity
if "\\input{tables/tab_phi_sg}" not in s:
    g = "\t\\input{tables/tab_gamma}\n"
    assert g in s, "gamma input anchor missing"
    s = s.replace(g, g + "\t\\input{tables/tab_phi_sg}\n", 1)

PHI = """\tTable~\\ref{tab:phisg} reports the sensitivity of the two free constants on the
\theld-out validation subset. $\\phi$ changes the factorised gate by at most $0.010$
\tover its full range, with the best value at the default $0.5$; $s_g$ matters more,
\tand its best value is $8$\\,px rather than the default $16$\\,px---another instance
\tof the strength effect, since a smaller saturation scale increases the effective
\tcorrection. Neither constant was tuned on evaluation data, and we do not claim
\tthat the defaults are globally optimal.

"""
fac = "\t\\subsection{The factorisation does not hold}"
if "tab:phisg" not in s:
    assert fac in s, "factorisation subsection anchor missing"
    s = s.replace(fac, PHI + fac, 1)

# 3) author contributions with unambiguous names
CONTRIB = """\t\\authorcontributions{Conceptualization, Rongchao Wang and Zhaofa Zhou;
\t\tmethodology, Rongchao Wang; software, Rongchao Wang and Zhaofa Zhou;
\t\tvalidation, all authors; formal analysis, Rongchao Wang; writing---original
\t\tdraft preparation, Rongchao Wang; writing---review and editing, Zhaofa Zhou
\t\tand Zhili Zhang; supervision, Zhili Zhang. All authors have read and agreed
\t\tto the published version of the manuscript. \\textbf{[Authors: please confirm
\t\tthe individual contributions; the previous ``Z.Z.'' abbreviation was ambiguous
\t\tbetween Zhaofa Zhou and Zhili Zhang and has been expanded as stated here.]}}"""
m = re.search("\t\\\\authorcontributions\\{.*?\\}\\s*\\n", s, flags=re.S)
if m and "Rongchao Wang and Zhaofa Zhou" not in m.group(0):
    s = s[:m.start()] + CONTRIB + s[m.end():]

# 4) equal-information-access comparison: report BOTH metrics with the paired CI.
FAIR_OLD_RE = re.compile(
    r"\tThe published-method comparison is informational.*?not as a state-of-the-art claim\.",
    flags=re.S)
FAIR_NEW = (
 "\tThe published-method comparison must be read with its information structure in\n"
 "\tmind, and its outcome depends on the metric. When each listed method receives the\n"
 "\toracle-registered infrared image---the same geometric information as the proposed\n"
 "\tcorrection---the strongest listed fusion method (SAGE-IVIF) reaches $0.6420$ mAP50\n"
 "\tand the proposed gate reaches $0.6709$, i.e.\ $+0.0289$ in favour of the gate on the\n"
 "\tdataset-level metric. On the per-image AP50 metric, computed with the same detector,\n"
 "\tthe same labels and the same 1049 image IDs, the same comparison reverses:\n"
 "\t$0.7344$ versus $0.7525$, i.e.\ $-0.0180$ with a paired image-level bootstrap $95\\,\\%$ CI of\n"
 "\t$[-0.0257,-0.0104]$. We therefore do not claim that the gate is better than a strong\n"
 "\tfusion method given equal geometric information; the two metrics disagree, and both\n"
 "\tare reported. The previously reported $+0.1163$ compared a corrected pipeline with\n"
 "\tuncorrected competitors and is withdrawn. We also note that the frozen detector was\n"
 "\ttrained on fused images produced by this project, which flatters pixel-average-like\n"
 "\tinputs; the table is a controlled ablation, not a state-of-the-art ranking.\n")
if FAIR_OLD_RE.search(s):
    s = FAIR_OLD_RE.sub(lambda _m: FAIR_NEW, s, count=1)
else:
    print("WARNING: fairness paragraph not matched")


# 5) learned fusion backbone (R1.6): table + paragraph after the quality-metric block
BACKBONE = (
 "\t\input{tables/tab_learned_backbone}\n"
 "\t\n"
 "\tTo check whether these conclusions depend on the deterministic pixel-average fusion, we\n"
 "\trepeat the four conditions on the outputs of a learned backbone (SeAFusion) on 300\n"
 "\tpairs (Table~\\ref{tab:backbone}). The ordering is unchanged: dense correction of the\n"
 "\tregistered inputs reaches $0.7710$, the factorised gate $0.7423$, the matched uniform\n"
 "\tgate $0.6963$ and no correction $0.6540$. The gate therefore recovers about three\n"
 "\tquarters of the available gain and beats the matched uniform gate by $0.046$, but it is\n"
 "\tstill below dense correction---the same pattern as on the pixel-average baseline. Note\n"
 "\tthat SeAFusion was trained on aligned data, so its registered-input condition is also\n"
 "\tits training regime; the comparison is therefore favourable to the dense condition.\n"
 "\n")
if "\\input{tables/tab_learned_backbone}" not in s:
    anchor = "\t\input{tables/tab_llvip}"
    assert anchor in s, "llvip input anchor missing"
    s = s.replace(anchor, BACKBONE + anchor, 1)


# 6) layered-parallax diagnostic (R1.3) + sweep CIs (R1.m5)
PARALLAX = (
 "\t\input{tables/tab_nonrigid}\n"
 "\t\n"
 "\tTo test the protocol dependence that reviewers asked about, we replace the global\n"
 "\taffine warp by a layered-parallax warp: three smooth depth layers with independent\n"
 "\ttranslations plus a smooth elastic residual, giving a spatially varying field whose\n"
 "\tground-truth correction displacement is still analytic (mean $4.5$\\,px; Table~\\ref{tab:nonrigid}).\n"
 "\tThe ordering of the affine protocol is preserved---full correction $0.7813$, task-only\n"
 "\t$0.7804$, factorised $0.7733$, geometry-only $0.7611$, uniform $0.7433$, none $0.7337$---so\n"
 "\tthe conclusions are not an artefact of a low-dimensional affine field. Two caveats\n"
 "\tremain: the layers are synthetic and are not derived from a depth sensor, and real\n"
 "\tparallax also brings occlusion, which this diagnostic does not model.\n"
 "\n")
if "\\input{tables/tab_nonrigid}" not in s:
    anchor = "\t\input{tables/tab_sweep}"
    assert anchor in s, "sweep input anchor missing"
    s = s.replace(anchor, anchor + "\n" + PARALLAX, 1)

# limitation sentence: the parallax diagnostic now exists, the real data still does not
LIM_OLD = ("The perturbation is a global affine\n"
           "\ttransform, so the results do not establish what happens under depth-dependent\n"
           "\tparallax, local non-rigid distortion, or occlusion;")
LIM_NEW = ("The main protocol is a global affine transform; we add a layered-parallax\n"
           "\tdiagnostic whose ordering matches, but real parallax data, a depth estimator and\n"
           "\tocclusion are still absent, so the results do not establish behaviour on real\n"
           "\tnon-rigid scenes;")
if LIM_OLD in s:
    s = s.replace(LIM_OLD, LIM_NEW, 1)
else:
    print("WARNING: limitation sentence not matched")


# 7) seed spreads for LLVIP and the parallax diagnostic (aggregated in exp/tables/seed_aggregate.json)
SEED_OLD = ("estimator is trained in-domain and with a single seed; the dataset therefore\n"
            "\tsupports the qualitative ordering, not a seed-level claim.")
SEED_NEW = ("estimator is trained in-domain; over three seeds ($\\rho=0.446,0.443,0.444$) the\n"
            "\tlearned gate gives $0.8529\\pm0.0008$, so the ordering is seed-stable and the gap to\n"
            "\tthe oracle task map ($0.9011$) is a prediction-quality effect rather than seed noise.")
if SEED_OLD in s:
    s = s.replace(SEED_OLD, SEED_NEW, 1)
else:
    print("WARNING: LLVIP single-seed sentence not matched")

LIM_OLD = "and the LLVIP experiment uses a single seed."
LIM_NEW = ("the LLVIP experiment now uses three seeds but remains a diagnostic on 800 pairs.")
if LIM_OLD in s:
    s = s.replace(LIM_OLD, LIM_NEW, 1)
else:
    print("WARNING: LLVIP limitation sentence not matched")

PAR_OLD = "\tparallax also brings occlusion, which this diagnostic does not model.\n"
PAR_NEW = ("\tparallax also brings occlusion, which this diagnostic does not model. Over three\n"
           "\ttraining seeds the factorised gate gives $0.7737\\pm0.0004$, the task-only gate\n"
           "\t$0.7796\\pm0.0007$ and the matched uniform gate $0.7440\\pm0.0013$, so the ordering and\n"
           "\tthe margins are seed-stable.\n")
if PAR_OLD in s:
    s = s.replace(PAR_OLD, PAR_NEW, 1)
else:
    print("WARNING: parallax caveat sentence not matched")


# 8) MS2 real parallax (R1.3): table + paragraph after the layered-parallax diagnostic
MS2 = (
 "\t\input{tables/tab_ms2}\n"
 "\t\n"
 "\tThe same question can be answered on data where the parallax is measured rather than\n"
 "\tsimulated. We use 300 frames of the MS2 multi-spectral stereo dataset, for which the\n"
 "\tRGB and thermal cameras share a calibrated rig and both have LiDAR depth projected\n"
 "\tinto their own frame; the RGB--thermal correspondence derived from that geometry is\n"
 "\tconsistent to $0.31$\\,px median round-trip error and gives a real parallax field with a\n"
 "\tmedian of $25.5$\\,px per frame (95th percentile $75.9$\\,px). We compare the pixel\n"
 "\taverage built from no registration, from a \\emph{global} model applied densely, from the\n"
 "\tsame model applied only where the parallax residual is small (a gated application), and\n"
 "\tfrom the per-pixel depth warp (Table~\\ref{tab:ms2}). The global model improves\n"
 "\tcross-modal mutual information in both the near and the far stratum, so applying it\n"
 "\tdensely ($0.4193$) beats every gated application (best $0.4057$), and the per-pixel\n"
 "\twarp is best ($0.4399$). Under real parallax the operative lever is therefore again the\n"
 "\tquality of the registration model rather than the spatial allocation of its output.\n"
 "\n")
if "\\input{tables/tab_ms2}" not in s:
    anchor = "\t\input{tables/tab_llvip}"
    assert anchor in s, "llvip anchor missing"
    s = s.replace(anchor, MS2 + anchor, 1)

# 9) one abstract sentence for the real-parallax evidence
ABS_OLD = "is at most $+0.017$. Separately, the optimal correction strength depends on the"
ABS_NEW = ("is at most $+0.017$. On measured RGB--thermal parallax (MS2, median $25.5$\\,px per\n"
           "\tframe) the same ordering holds under cross-modal mutual information: a global model\n"
           "\tapplied densely ($0.4193$) beats its gated application ($0.4057$), and the per-pixel\n"
           "\tdepth warp is best ($0.4399$). Separately, the optimal correction strength depends on the")
if ABS_OLD in s and "MS2, median" not in s:
    s = s.replace(ABS_OLD, ABS_NEW, 1)
elif "MS2, median" not in s:
    print("WARNING: abstract anchor not matched")


# 10) VTMOT detection-level real-misalignment check
VTMOT = (
 "\t\input{tables/tab_vtmot}\n"
 "\t\n"
 "\tBecause the MS2 package has no detection labels, we add a detection-level check on\n"
 "\treal misalignment using six VTMOT sequences (240 frames) with MOT boxes. The two\n"
 "\tcameras are natively misaligned by $7$--$21$\\,px; a per-sequence affine estimated with\n"
 "\tECC improves cross-modal MI by $+0.013$ to $+0.054$ on every sequence, so the\n"
 "\tregistration is genuinely correcting the misalignment. Detection is evaluated with a\n"
 "\tCOCO detector (pedestrian and car) and paired per-image bootstraps. The result is a\n"
 "\tnull effect: $0.1984$ (none), $0.1933$ (dense), $0.1990$ (gated), $0.1951$ (uniform),\n"
 "\twith none$-$dense $=-0.0026$ $[-0.0148,0.0100]$ and dense$-$gated $=-0.0009$\n"
 "\t$[-0.0123,0.0108]$. Two readings follow. The detector used here is trained on clean\n"
 "\timagery and is insensitive to the ghosting produced by a $7$--$21$\\,px offset on\n"
 "\tsmall UAV objects, so detection-level sensitivity to misalignment is a property of the\n"
 "\tdetector as much as of the geometry---the frozen fused-image detector of the main\n"
 "\texperiments is far more sensitive. And even in this regime the gated application is\n"
 "\tindistinguishable from the dense one, so allocation does not become the lever.\n"
 "\n")
if "\\input{tables/tab_vtmot}" not in s:
    anchor = "\t\input{tables/tab_llvip}"
    assert anchor in s, "llvip anchor for vtmot missing"
    s = s.replace(anchor, VTMOT + anchor, 1)

# 11) two residual reviewer points: R1.m4 (relative normalisation) and R1.m6 (pipeline difference)
NORM_OLD = ("taken once per image.\n")
NORM_NEW = ("taken once per image. The relative, per-image normalisation $\\mathcal{N}(\\cdot)$ is\n"
            "\tthe mechanism behind the failure at small magnitudes: because it rescales each image's\n"
            "\tdisplacement field to its own maximum, a uniformly small misalignment still produces\n"
            "\tlarge gate values, so the gate cannot decline when correction is harmful. A gate\n"
            "\texpressed in absolute pixels would need $s_g$ to be calibrated per magnitude, which we\n"
            "\tdo not claim.\n")
if "the mechanism behind the failure at small magnitudes" not in s:
    assert NORM_OLD in s
    s = s.replace(NORM_OLD, NORM_NEW, 1)

PIPE_OLD = ("The previously reported $+0.1163$ compared a corrected pipeline with\n"
            "\tuncorrected competitors and is withdrawn.")
PIPE_NEW = ("The no-correction reference differs slightly from the main table ($0.5891$ here\n"
            "\tversus $0.5887$ in Table~\\ref{tab:gates}) because this runner resizes to\n"
            "\t$480\\times640$ and applies its own pre-processing before fusion; the difference is\n"
            "\tpipeline-level and does not affect the within-table ranking. The previously reported\n"
            "\t$+0.1163$ compared a corrected pipeline with uncorrected competitors and is withdrawn.")
if "pipeline-level and does not affect" not in s:
    assert PIPE_OLD in s, "fairness pipeline anchor missing"
    s = s.replace(PIPE_OLD, PIPE_NEW, 1)


# 12) second real registration module (SIFT+RANSAC)
SIFT = (
 "\t\input{tables/tab_e2_sift}\n"
 "\t\n"
 "\tTo check that this result is not a property of one estimator, we repeat the\n"
 "\tend-to-end experiment with a structurally different module: SIFT keypoints on\n"
 "\tgradient-magnitude images, ratio-test matching, RANSAC partial-affine estimation and a\n"
 "\tscale sanity check. It converges on $41\\,\\%$ of the pairs (median corner error\n"
 "\t$5.0$\\,px, displacement error $3.3$\\,px on those) and abstains otherwise, in which\n"
 "\tcase no correction is applied. Two readings follow. First, the failure mode moves from\n"
 "\tthe allocation to the candidate: on the 59\\,\\% of pairs where the matcher abstains the\n"
 "\tgate cannot recover the oracle gain, and the deployment configuration (C3, $0.6099$)\n"
 "\tsits only slightly above no correction. Second, the ordering is preserved: dense\n"
 "\tapplication of the estimated warp (C4, $0.6156$) is at least as good as its gated\n"
 "\tapplication (C3, $0.6099$). Both modules therefore point at the same lever, the quality\n"
 "\tof the registration candidate, rather than at the spatial allocation of its output.\n"
 "\n")
if "\\input{tables/tab_e2_sift}" not in s:
    anchor = "\t\input{tables/tab_e2_sweep}"
    assert anchor in s, "e2sweep anchor missing"
    s = s.replace(anchor, SIFT + anchor, 1)

# 13) final author/availability decisions: initials, real repository URL, no AI statement
import re as _re

DA_OLD = _re.compile(r"\t\\dataavailability\{.*?acceptance\.\}", _re.S)
DA_NEW = ("\t\\dataavailability{The M3FD and LLVIP datasets are publicly available (see\n"
          "\t\treferences). The experiment code, the eight trained checkpoints, the evaluation\n"
          "\t\tmanifests, the controlled-misalignment parameters, the per-image predictions, the\n"
          "\t\tfusion-metric implementation, the table builders and a single-command reproduction\n"
          "\t\tfile are available at \\url{https://github.com/WRCDEBUG/Registration-Necessity-Guided};\n"
          "\t\tSHA256 checksums are included and every number in the paper is indexed in a results\n"
          "\t\tledger.}")
if DA_OLD.search(s):
    s = DA_OLD.sub(lambda _m: DA_NEW, s, count=1)
else:
    print("WARNING: data availability block not matched")

ACK_OLD = _re.compile(r"\t\\acknowledgments\{.*?publication\.\}\n", _re.S)
if ACK_OLD.search(s):
    s = ACK_OLD.sub("", s, count=1)
else:
    print("WARNING: acknowledgments block not matched")

CONTRIB_NEW = ("\t\\authorcontributions{Conceptualization, R.W. and Z.Z.L.; methodology, R.W.;\n"
               "\t\tsoftware, R.W. and Z.Z.F.; validation, all authors; formal analysis, R.W.;\n"
               "\t\twriting---original draft preparation, R.W.; writing---review and editing, Z.Z.F.\n"
               "\t\tand Z.Z.L.; supervision, Z.Z.L. All authors have read and agreed to the published\n"
               "\t\tversion of the manuscript.}")
M = _re.search(r"\t\\authorcontributions\{.*?\}\n", s, _re.S)
if M and "Z.Z.L." not in M.group(0):
    s = s[:M.start()] + CONTRIB_NEW + s[M.end():]
elif not M:
    print("WARNING: contributions block not matched")
s = s.replace("released with the anonymised repository",
              "released with the repository listed in the data-availability statement")


# 14) keep the RN-BC name, with one sentence explaining what "necessity" means here
NAME_OLD = ("Figure~\\ref{fig:framework} separates the proposed framework into three")
NAME_NEW = ("We keep the name RN-BC (registration-necessity-guided bounded correction) for\n"
            "\tcontinuity with the reviewed version, but the study changes what \\emph{necessity}\n"
            "\tmeans in it: the operative quantity is the global magnitude of the correction, not the\n"
            "\tspatial allocation of its output. The name therefore describes the correction rule,\n"
            "\tnot a claim that spatial gating is the better policy.\n"
            "\n"
            "Figure~\\ref{fig:framework} separates the proposed framework into three")
if "We keep the name RN-BC" not in s:
    assert NAME_OLD in s, "method-name anchor missing"
    s = s.replace(NAME_OLD, NAME_NEW, 1)

DST.write_text(s)
print("wrote", DST.name, len(s.splitlines()), "lines")
print("url:", "github.com/WRCDEBUG" in s, "| ack removed:", "\\acknowledgments" not in s)
print("re-added:", "provenance" if "Downstream detector provenance" in s else "MISSING",
      "| phi-sg:", "\\input{tables/tab_phi_sg}" in s,
      "| contributions:", "Z.Z.L." in s and "Z.Z.F." in s,
      "| CI:", "paired image-level bootstrap" in s)
