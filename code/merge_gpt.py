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


# 15) explicit statement that earlier studies were not reproduced (R1.m9)
REP_OLD = ("Correlation with geometry is therefore not a proxy for\n"
           "\tdownstream utility.")
REP_NEW = ("Correlation with geometry is therefore not a proxy for\n"
           "\tdownstream utility. We also did not reproduce the datasets used by the earlier\n"
           "\tproxy-supervised study: its released map is used here only as a baseline under our\n"
           "\tprotocol, so we make no claim about why earlier studies reported the outcomes they\n"
           "\tdid, and the unsupported statements in the original manuscript have been removed.")
if "did not reproduce the datasets" not in s:
    assert REP_OLD in s, "discussion anchor missing"
    s = s.replace(REP_OLD, REP_NEW, 1)


# 16) round-2 item 1: reframe as protocol + negative result, drop the method name
TITLE_OLD = ("\\Title{Registration Necessity Is a Magnitude Decision: A Controlled Study of\n"
             "\tSpatially Selective Correction for Misaligned Infrared--Visible Fusion}")
TITLE_NEW = ("\\Title{Registration Necessity Is a Magnitude Decision: A Protocol and Negative\n"
             "\tResult for Spatially Selective Correction in Misaligned Infrared--Visible Fusion}")
if TITLE_OLD in s:
    s = s.replace(TITLE_OLD, TITLE_NEW, 1)
else:
    print("WARNING: title anchor not matched")

ABS_OLD = ("spatially using a ``registration necessity'' map. We test that premise with a\n"
           "\tprotocol in which both factors of necessity are measurable:")
ABS_NEW = ("spatially using a ``registration necessity'' map. We report a \\emph{negative result}\n"
           "\tfor that premise, obtained with a protocol in which both factors of necessity are\n"
           "\tmeasurable:")
if ABS_OLD in s:
    s = s.replace(ABS_OLD, ABS_NEW, 1)
else:
    print("WARNING: abstract anchor not matched")

NAME_OLD = ("We keep the name RN-BC (registration-necessity-guided bounded correction) for\n"
            "\tcontinuity with the reviewed version, but the study changes what \\emph{necessity}\n"
            "\tmeans in it: the operative quantity is the global magnitude of the correction, not the\n"
            "\tspatial allocation of its output. The name therefore describes the correction rule,\n"
            "\tnot a claim that spatial gating is the better policy.")
NAME_NEW = ("This section describes the design that is \\emph{tested and rejected} in this paper. We\n"
            "\tno longer present it as a proposed method and we do not recommend it for deployment:\n"
            "\tits gated configuration is beaten by dense correction at every magnitude above\n"
            "\t$\\pm8$\\,px, by a zero-parameter displacement threshold, and by the task map alone,\n"
            "\tand the multiplicative factorisation is not supported. The released code keeps the\n"
            "\tinternal name RN-BC used during development so that the artefacts remain traceable.")
if NAME_OLD in s:
    s = s.replace(NAME_OLD, NAME_NEW, 1)
else:
    print("WARNING: naming paragraph not matched")

s = s.replace("\\section{Proposed Method}", "\\section{Tested Design and Measurement Protocol}", 1)
s = s.replace("bounded correction (RN-BC). Unless stated otherwise we use the YOLOv8n checkpoint\n"
              "\tdistributed with the reported RN-BC experiments",
              "bounded correction. Unless stated otherwise we use the YOLOv8n checkpoint\n"
              "\tdistributed with the reported experiments", 1)
s = s.replace("RN-BC itself does not introduce an additional learned", "This design does not introduce an additional learned", 1)
s = s.replace("the computational cost reported for RN-BC refers to the", "the computational cost reported for this design refers to the", 1)
s = s.replace("is not jointly trained with RN-BC.", "is not jointly trained with this design.", 1)
s = s.replace("\t\t\tRN-BC & Registration-necessity-guided bounded correction \\\\\n", "", 1)
s = s.replace("RN-BC & Registration-necessity-guided bounded correction \\\\\n", "", 1)


# 17) round-2 item 1: state the negative result in the contribution list and the outline
CONTR_OLD = "The contributions of this paper are summarized as follows."
CONTR_NEW = ("The contribution of this paper is primarily a \\emph{negative} one: a tested,\n"
             "\treproducible protocol under which spatially selective correction does not deliver what\n"
             "\tthe current literature expects from it, together with the positive finding that the\n"
             "\tdecisive quantity is the magnitude of the correction rather than its allocation. The\n"
             "\tindividual contributions are as follows.")
if "primarily a \\emph{negative} one" not in s:
    s = s.replace(CONTR_OLD, CONTR_NEW, 1)

ITEM_OLD = ("\\item We show that necessity is a \\emph{magnitude} decision: at $\\pm2$\\,px\n"
            "\t\tcorrection is harmful, image-level gating degenerates to ``always correct'' on\n"
            "\t\tour test set")
ITEM_NEW = ("\\item We show that necessity is a \\emph{magnitude} decision: at $\\pm2$\\,px\n"
            "\t\tcorrection is harmful, image-level gating degenerates to ``always correct'' on\n"
            "\t\tour test set, and on a full-size set with a controlled mixture of magnitudes the\n"
            "\t\tper-image decision rule is tested against always-correct with a paired interval")
if ITEM_OLD in s:
    s = s.replace(ITEM_OLD, ITEM_NEW, 1)

OUT_OLD = "Section~\\ref{sec:method} presents the measurable necessity\n\tdefinition, the estimator, and the bounded correction rule."
OUT_NEW = ("Section~\\ref{sec:method} defines the tested design and the measurement\n"
           "\tprotocol, and states why it is not recommended.")
if OUT_OLD in s:
    s = s.replace(OUT_OLD, OUT_NEW, 1)


# 18) round-2 item 2: the mixed-magnitude result contradicts the magnitude-decision title
T2_OLD = ("\\Title{Registration Necessity Is a Magnitude Decision: A Protocol and Negative\n"
          "\tResult for Spatially Selective Correction in Misaligned Infrared--Visible Fusion}")
T2_NEW = ("\\Title{Registration Necessity Is Not a Spatial-Allocation Problem: A Protocol and\n"
          "\tNegative Result for Selective Correction in Misaligned Infrared--Visible Fusion}")
if T2_OLD in s:
    s = s.replace(T2_OLD, T2_NEW, 1)

ABS_MAG_OLD = ("Separately, the optimal correction strength depends on the")
ABS_MAG_NEW = ("A controlled mixture of magnitudes on the full evaluation set (a quarter of the pairs\n"
               "\tbelow $4$\\,px) shows that the harmful regime is confined to $<2$\\,px, that correction\n"
               "\talready pays at $2$--$4$\\,px, and that no magnitude-threshold rule beats\n"
               "\talways-correct, so the decision ``correct or not'' does not survive as a policy. "
               "Separately, the optimal correction strength depends on the")
if "the harmful regime is confined to" not in s:
    s = s.replace(ABS_MAG_OLD, ABS_MAG_NEW, 1)


# 19) round-2 item 2: mixed-magnitude section
MIXED = (
 "\t\\input{tables/tab_mixed}\n"
 "\t\n"
 "\tThe magnitude evidence so far came from a 300-pair subset, and the main test set has no\n"
 "\tsmall-misalignment subgroup (10th percentile of the mean displacement $5.85$\\,px), so the\n"
 "\tclaim that necessity is a magnitude decision was not tested where it matters. We rebuilt\n"
 "\tthe evaluation set on all 1049 pairs with a controlled mixture of magnitudes---25\\,\\% of\n"
 "\tthe pairs below $4$\\,px, median $8.94$\\,px---and re-ran no correction and full correction\n"
 "\t(Table~\\ref{tab:mixed}). Two findings follow. First, the harmful regime is much narrower\n"
 "\tthan the synthetic sweep suggested: correction is harmful only below $<2$\\,px\n"
 "\t($-0.0151$), and it already pays at $2$--$4$\\,px ($+0.0265$, CI $[+0.0152,+0.0391]$).\n"
 "\tSecond, no per-image rule that corrects only above a threshold $\\tau$ beats\n"
 "\talways-correct: $\\tau=4$\\,px costs $-0.0058$ (CI $[-0.0088,-0.0030]$), and $\\tau=2$\\,px is\n"
 "\tindistinguishable from zero ($+0.0003$, CI $[0.0000,+0.0008]$). The decision ``correct or\n"
 "\tnot'' therefore does not survive as a policy: in practice one should correct, densely.\n"
 "\n")
if "\\input{tables/tab_mixed}" not in s:
    anchor = "\t\\input{tables/tab_nonrigid}"
    assert anchor in s, "nonrigid anchor missing"
    s = s.replace(anchor, MIXED + anchor, 1)


# 21) round-2 item 5: explain the metric divergence and declare the primary metric
DIV_OLD = ("The no-correction reference differs slightly from the main table")
DIV_NEW = ("The two metrics disagree, and we do not attribute the disagreement to a single stratum.\n"
           "\tSplitting the same 1049 images by the number of annotated objects gives ours$-$SAGE\n"
           "\t$=+0.017$ (one box, $n=91$), $-0.028$ (two or three, $n=215$) and $-0.019$ (four or\n"
           "\tmore, $n=743$); splitting by mean object area gives $-0.019$, $-0.011$ and $-0.024$ for\n"
           "\tthe small, medium and large terciles; and the share of low-confidence detections\n"
           "\t($\\mathrm{conf}<0.25$) is $84.3\\,\\%$ versus $85.9\\,\\%$. The divergence is therefore a\n"
           "\tproperty of the aggregation rather than of a region or a size class: dataset-level\n"
           "\tmAP50 pools one precision--recall curve per class over all images, so it is sensitive to\n"
           "\thow detections redistribute along the confidence axis, whereas per-image AP50 computes a\n"
           "\tten-point interpolated curve for every image and averages, weighting each image equally,\n"
           "\tincluding the 91 single-object images. We report dataset-level mAP50 as the primary\n"
           "\tmetric because it is the metric used for every comparison in this paper and in the\n"
           "\tfusion literature, and it is the metric on which the main claims are made; per-image\n"
           "\tAP50 is kept as a paired diagnostic that supports the bootstrap intervals. Neither\n"
           "\tmetric is used to claim superiority over the published methods.\n"
           "\n"
           "\tThe no-correction reference differs slightly from the main table")
if "we do not attribute the disagreement to a single stratum" not in s:
    assert DIV_OLD in s, "fairness anchor missing"
    s = s.replace(DIV_OLD, DIV_NEW, 1)


# 22) round-2 item 4: detector bias (three detectors)
DETB = (
 "\t\\input{tables/tab_detector_bias}\n"
 "\t\n"
 "\tThe frozen detector used in the rest of the paper was trained on fused images produced by\n"
 "\tthis project and its checkpoint was selected on the split that is later evaluated, both of\n"
 "\twhich favour pixel-average-like inputs. To test whether the conclusions depend on that\n"
 "\tchoice we retrained YOLOv8n twice from COCO weights on a different fusion source---SAGE-fused\n"
 "\ttraining pairs and raw visible images---with a train/validation split inside the training\n"
 "\tpool, so that the 1049 evaluation pairs are never used for checkpoint selection. The\n"
 "\tordering is preserved under all three detectors (Table~\\ref{tab:detbias}): full correction\n"
 "\tfirst, the displacement threshold next, the task map and the factorised gate close together,\n"
 "\tand the uniform gate above no correction. The absolute benefit of correction, however, is\n"
 "\tnot preserved: it falls from $+0.104$ with the original detector to $+0.083$ with the\n"
 "\tSAGE-trained detector and to $+0.021$ with the visible-only detector. The ranking claims\n"
 "\ttherefore hold, but the size of the effect is partly a property of the detector, and we\n"
 "\treport the three detectors rather than the most favourable one.\n"
 "\n")
if "\\input{tables/tab_detector_bias}" not in s:
    anchor = "\t\\input{tables/tab_mixed}"
    assert anchor in s, "mixed anchor missing"
    s = s.replace(anchor, DETB + anchor, 1)


# 23) acronym-first title in the style of a Q2 journal (names the PROTOCOL, not a method)
T3_OLD = ("\\Title{Registration Necessity Is Not a Spatial-Allocation Problem: A Protocol and\n"
          "\tNegative Result for Selective Correction in Misaligned Infrared--Visible Fusion}")
T3_NEW = ("\\Title{SPARC: A Protocol and Negative Result on the Spatial Allocation of\n"
          "\tRegistration Correction in Misaligned Infrared--Visible Fusion}")
if T3_OLD in s:
    s = s.replace(T3_OLD, T3_NEW, 1)
else:
    print("WARNING: acronym title anchor not matched")

SPARC_OLD = ("We report a \\emph{negative result}\n"
             "\tfor that premise, obtained with a protocol in which both factors of necessity are\n"
             "\tmeasurable:")
SPARC_NEW = ("We report a \\emph{negative result}\n"
             "\tfor that premise, obtained with SPARC (Spatial Allocation of Registration Correction),\n"
             "\ta protocol in which both factors of necessity are measurable:")
if SPARC_OLD in s:
    s = s.replace(SPARC_OLD, SPARC_NEW, 1)
else:
    print("WARNING: abstract SPARC anchor not matched")


# 24) remove the two remaining statements that contradict the mixed-magnitude result
ABS_TAIL_OLD = ("Separately, the optimal correction strength depends on the\n"
                "\t\\emph{global} misalignment magnitude---at $\\pm2$\\,px correction is harmful---which\n"
                "\tmakes necessity a magnitude decision rather than a spatial-allocation problem. We\n"
                "\trelease the protocol, all gate baselines and per-image predictions.")
ABS_TAIL_NEW = ("Separately, the optimal correction strength depends on the global\n"
                "\tmisalignment magnitude, but that dependence does not produce a better policy than\n"
                "\tcorrecting densely. We release the protocol, all gate baselines and per-image\n"
                "\tpredictions.")
if ABS_TAIL_OLD in s:
    s = s.replace(ABS_TAIL_OLD, ABS_TAIL_NEW, 1)
else:
    print("WARNING: abstract tail anchor not matched")

ITEM_OLD = ("\\item We show that necessity is a \\emph{magnitude} decision: at $\\pm2$\\,px\n"
            "\t\tcorrection is harmful, image-level gating degenerates to ``always correct'' on\n"
            "\t\tour test set, and on a full-size set with a controlled mixture of magnitudes the\n"
            "\t\tper-image decision rule is tested against always-correct with a paired interval, and "
            "fusion-quality metrics only predict downstream performance when\n"
            "\t\tthe reference is the clean scene rather than the observed misaligned input.")
ITEM_NEW = ("\\item We test the magnitude argument where it matters: on a full-size set with a\n"
            "\t\tcontrolled mixture of magnitudes (a quarter of the pairs below $4$\\,px) the harmful\n"
            "\t\tregime is confined to $<2$\\,px, correction already pays at $2$--$4$\\,px, and no\n"
            "\t\tthreshold rule beats always-correct, so image-level gating degenerates to ``always\n"
            "\t\tcorrect'' in practice. Fusion-quality metrics predict downstream performance only\n"
            "\t\twhen the reference is the clean scene rather than the observed misaligned input.")
if ITEM_OLD in s:
    s = s.replace(ITEM_OLD, ITEM_NEW, 1)
else:
    print("WARNING: contribution item anchor not matched")


# 25) round-2 item 3: learned registrar (RAFT) table + paragraph
RAFT = (
 "\t\\input{tables/tab_e2_raft}\n"
 "\t\n"
 "\tThe two modules above are classical. We therefore add a learned one: pretrained RAFT optical\n"
 "\tflow on the misaligned pair, converted to correspondences on a coarse grid and reduced to a\n"
 "\tpartial affine with RANSAC (Table~\\ref{tab:e2raft}). It converges on $99.9\\,\\%$ of the 1049\n"
 "\tpairs with a median corner error of $7.2$\\,px and a displacement error of $5.1$\\,px---better\n"
 "\tthan the gradient-magnitude search ($8.1$\\,px) and with far wider coverage than SIFT\n"
 "\t($41\\,\\%$). With this module the deployment configuration reaches $0.6267$ and dense\n"
 "\tapplication of the same estimate $0.6337$, so dense correction is again at least as good as\n"
 "\tits gated version, and the registration-quality conclusion holds for a learned registrar and\n"
 "\tnot only for classical ones.\n"
 "\n")
if "\\input{tables/tab_e2_raft}" not in s:
    anchor = "\t\\input{tables/tab_e2_sweep}"
    assert anchor in s, "e2sweep anchor missing"
    s = s.replace(anchor, RAFT + anchor, 1)

DST.write_text(s)
print("wrote", DST.name, len(s.splitlines()), "lines")
print("url:", "github.com/WRCDEBUG" in s, "| ack removed:", "\\acknowledgments" not in s)
print("re-added:", "provenance" if "Downstream detector provenance" in s else "MISSING",
      "| phi-sg:", "\\input{tables/tab_phi_sg}" in s,
      "| contributions:", "Z.Z.L." in s and "Z.Z.F." in s,
      "| CI:", "paired image-level bootstrap" in s)
