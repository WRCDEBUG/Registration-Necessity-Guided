#!/usr/bin/env python3
"""Build the route-A revision of the manuscript from the submitted version.

Strategy: keep the method section and the measured-protocol description (still
accurate), replace the framing (title/abstract/contributions) and rebuild the
results/discussion/conclusion around the evidence chain in exp/E1_findings.md,
exp/E2_findings.md and exp/E3_findings.md.

The original main.tex is never modified; the output is a separate file.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SENSE = HERE.parent
SRC = SENSE / "main.tex"
DST = SENSE / "main_revision_A.tex"

src = SRC.read_text()

TITLE = r"""\Title{Registration Necessity Is a Magnitude Decision: A Controlled Study of
Spatially Selective Correction for Misaligned Infrared--Visible Fusion}"""

ABSTRACT = r"""\abstract{Misaligned infrared--visible fusion is usually handled by dense
registration, and a recent line of work instead \emph{gates} the correction
spatially using a ``registration necessity'' map. We test that premise with a
protocol in which both factors of necessity are measurable: the perturbation is
analytic, so the per-pixel correction demand is exact, and task importance is
rasterised from detection annotations. Across 1049 M3FD pairs and three training
seeds the premise does not hold under controlled affine misalignment. Dense
correction of the registered candidate is the strongest policy (mAP50 $0.6922$
versus $0.5887$ without correction), and every gate is below it: a zero-parameter
displacement threshold reaches $0.6847$, a learned task-importance map $0.6732$,
the factorised geometry\,$\times$\,task map $0.6709$, and a matched uniform gate
$0.6380$. Two structural results explain the ordering: multiplying by geometric
demand \emph{reduces} accuracy relative to the task map alone (paired
$\Delta=+0.0191$, $95\,\%$ CI $[+0.0139,+0.0243]$), and a proxy map from earlier
work is not worse than our learned map when used as a gate ($0.6792$). With a real
cross-modal registration module (median corner error $8.1$\,px) the correction
gain collapses from $+0.099$ to $+0.018$, registration error outweighs
geometric-factor error by $15\times$, and gating becomes statistically
indistinguishable from dense correction. A calibrated error sweep locates the
crossover: gating, even with oracle reliability, only helps once the registration
error exceeds $\approx12$\,px, where correction itself is worthless and its gain
is at most $+0.017$. Separately, the optimal correction strength depends on the
\emph{global} misalignment magnitude---at $\pm2$\,px correction is harmful---which
makes necessity a magnitude decision rather than a spatial-allocation problem. We
release the protocol, all gate baselines and per-image predictions.}"""

KEYWORDS = r"""\keyword{infrared--visible image fusion; misaligned multimodal perception;
registration necessity; selective registration; evaluation protocol; object detection}"""

CONTRIB = r"""The contributions of this paper are summarized as follows.

\begin{itemize}
\item We introduce a measured protocol in which both factors of registration
necessity are computable: the correction demand from an analytic perturbation and
the task importance from detection annotations. The resulting diagnostics are
pairwise reproducible, which makes gate-level claims falsifiable.
\item We compare twelve gate strategies under one detector, one perturbation set
and three training seeds. Dense correction dominates; a zero-parameter
displacement threshold saturates the achievable gain; a learned task map and a
matched uniform gate are both below it.
\item We show that the multiplicative factorisation is not supported by the data:
the task map alone is significantly better than the geometry\,$\times$\,task
product, and the same ordering holds at the oracle level in both datasets.
\item We replace the oracle geometry with a real cross-modal registration module,
separate geometric-factor error from warp error ($0.004$ versus $0.065$ in
per-image AP50), and locate the error range in which gating would pay off
($\approx12$\,px mean displacement error, with a gain of at most $+0.017$).
\item We show that necessity is a \emph{magnitude} decision: at $\pm2$\,px
correction is harmful, image-level gating degenerates to ``always correct'' on
our test set, and fusion-quality metrics only predict downstream performance when
the reference is the clean scene rather than the observed misaligned input.
\end{itemize}"""

RESULTS = r"""
\subsection{Gate maps under the controlled protocol}
\label{sec:gates}

Table~\ref{tab:gates} reports every gate on the 1049 evaluation pairs with
$\gamma=1$; Table~\ref{tab:seeds} gives the training-seed spread. Three facts
stand out.

First, \emph{dense correction is the strongest policy}: full correction reaches
$0.6922$, and no gate reaches it. Second, a \emph{zero-parameter displacement
threshold} ($\bar d>0.3$) reaches $0.6847$ and already exceeds every learned gate;
the margin to full correction is $0.0075$. Third, the learned gates themselves
order as $\text{task only} > \text{factorised} > \text{end-to-end}$, and the
matched uniform gate is far below all of them ($0.6380$). Spatial allocation
therefore matters relative to a uniform gate (paired per-image gain
$+0.0445$, $95\,\%$ CI $[+0.0360,+0.0534]$), but it is not where the main gain
comes from, and it is not what separates a good gate from a mediocre one.

\input{tables/tab_gate_matrix}
\input{tables/tab_seeds}

\subsection{The factorisation does not hold}
\label{sec:factorisation}

The factorised map multiplies a geometric-demand term by a task-importance term.
Its own oracle ladder contradicts the product: with an ideal task map, the
product ($0.6758$) is \emph{worse} than the task map alone ($0.6865$), and the
same holds for the learned maps ($0.6732$ versus $0.6709$; paired per-image
$\Delta=+0.0191$, $95\,\%$ CI $[+0.0139,+0.0243]$). The effect is not a training
artefact: it appears in all three seeds and on LLVIP
(Section~\ref{sec:llvip}).

Two mechanisms explain it. Geometrically, the multiplicative form compresses the
task term to the interval $[\phi,1]$, so the geometric factor dominates the output
range; perceptually, a soft gate produces fractional correction weights
($\gamma N\in(0,1)$), and a fractional blend of a misaligned pair keeps a
residual double edge. We tested the second mechanism directly: binarising the
task map or the product at $0.5$ does \emph{not} help ($0.6674$ and $0.6624$),
whereas a threshold on the geometric term does help ($0.6847$ versus $0.6689$ for
the soft form), because it increases the effective strength of the correction
rather than its sharpness. The operative variable is therefore how much
correction is applied, at what strength, not how the gate is shaped.

\subsection{Real registration and the boundary of the argument}
\label{sec:estimated}

All results above use the analytic transform as the registration candidate, which
is the main limitation of a controlled protocol. We therefore replace it with a
real cross-modal registrar: a bounded four-parameter search (rotation,
translation, scale) on gradient-magnitude images, with the search bounds set by
the acquisition prior rather than by the ground truth. Its accuracy is
$8.1$\,px median corner error and $3.2$\,px median displacement error, i.e.\ it
recovers roughly half of the induced misalignment---a realistic, not an
oracle, module.

\input{tables/tab_e2_real}

Table~\ref{tab:e2real} separates the two error channels. Feeding the estimated
geometric factor into the gate of an oracle warp costs only $0.0043$ per-image
AP50, whereas using the estimated warp costs $0.0645$: registration error
outweighs geometric-factor error by $15\times$, and the total cost of a real
pipeline is $0.0703$. Under the estimated warp, gating and dense correction are
statistically indistinguishable ($+0.0030$, $95\,\%$ CI $[-0.0024,+0.0086]$), and
both remain slightly above no correction ($+0.0147$ and $+0.0178$).

To locate where gating could ever pay off we scan a calibrated smooth error field
added to both the warp and the geometric factor (Table~\ref{tab:e2sweep}). For
mean displacement errors up to $8$\,px, dense correction is best and every gate
is worse---including a gate that uses the \emph{true} error to suppress
unreliable regions. The crossover appears only at $\approx12$\,px, at which point
dense correction has fallen to the no-correction level ($0.5877$ versus $0.5887$)
and the best gate is worth $+0.017$. In other words, spatial gating becomes
attractive only where the registration candidate is already worthless.

\input{tables/tab_e2_sweep}

\subsection{Magnitude, noise and fusion quality}
\label{sec:magnitude}

\input{tables/tab_sweep}

Table~\ref{tab:sweep} varies the misalignment magnitude. The key observation is
at the low end: for $\pm2$\,px, \emph{no correction} ($0.7875$) beats full
correction ($0.7809$), because resampling costs more than the residual
misalignment. From $\pm8$\,px upwards, full correction is best at every magnitude
and every gate stays below it, while the uniform gate degrades fastest. The
optimal correction strength is therefore a function of the \emph{global}
misalignment magnitude, and it is a decision of the form ``correct or not'',
taken once per image.

We tested that decision directly: selecting, per image, either the unregistered
or the fully corrected fused image according to the mean displacement gives
$0.6921$ at a threshold of $2$\,px, which corrects $1046$ of $1049$ images---the
image-level gate degenerates to ``always correct'' because the test set contains
no genuinely small-misalignment subgroup (the 10th percentile of the mean
displacement is $5.85$\,px). Additive input noise leaves the proposed gate
largely unaffected ($0.7681\to0.7618$ at $\sigma=0.08$) while the uniform gate is
insensitive by construction.

\input{tables/tab_fusion_quality}

Table~\ref{tab:fusionq} reports fusion-quality metrics. Their ordering depends
critically on the reference: measured against the \emph{observed} misaligned
infrared image, no correction scores best on MI, Qabf and VIF; measured against
the \emph{clean} scene, full correction scores best and the ordering
($\text{full}>\text{threshold}>\text{factorised}>\text{uniform}>\text{none}$)
agrees with detection. This is a methodological result in its own right: quality
metrics evaluated against the observed inputs systematically favour leaving the
misalignment uncorrected, which explains the recurring mismatch between
fusion-quality scores and downstream detection in this literature.

\subsection{Second dataset}
\label{sec:llvip}

\input{tables/tab_llvip}

On LLVIP (800 evaluation pairs, documented RT-DETR-L detector) the same structure
reappears (Table~\ref{tab:llvip}). Registration matters
($0.8121\to0.9009$), the oracle task map matches full correction ($0.9011$),
multiplying by geometric demand costs $0.039$ ($0.8622$), and the uniform gate is
\emph{worse} than no correction ($0.7803$ vs $0.8121$)---the same sign as the
originally reported per-image result ($-0.031$). The LLVIP task-importance
estimator is trained in-domain and with a single seed; the dataset therefore
supports the qualitative ordering, not a seed-level claim.

\subsection{Comparison under equal information access}
\label{sec:fair}

\input{tables/tab_fairness}

The published-method comparison is informational and must be read with its
information structure in mind. When each method receives the oracle-registered
infrared image---the same geometric information as the proposed correction---the
strongest fusion method reaches $0.6420$ and the proposed gate reaches $0.6709$,
i.e.\ $+0.0289$. The previously reported $+0.1163$ compared a corrected pipeline
with uncorrected competitors and is not a method comparison; it is replaced here.
We also note that the frozen detector was trained on fused images produced by this
project, which flatters pixel-average-like inputs; the ranking should therefore be
read as a controlled ablation, not as a state-of-the-art claim.

"""

DISCUSSION = r"""The experiments support five bounded conclusions. First, registration
necessity is measurable, and the measured version makes gate-level claims
falsifiable. Second, under controlled affine misalignment the dominant factor is
the \emph{amount and strength} of the correction, not its spatial allocation:
dense correction wins at every magnitude from $\pm8$\,px upwards, a zero-parameter
displacement threshold saturates the learned gate, and a matched uniform gate is
the weakest policy. Third, the multiplicative factorisation is not supported:
the task map alone is significantly better than the product, at both the learned
and the oracle level, and on both datasets. Fourth, registration quality dominates
everything that can be gained by gating: real cross-modal registration error costs
$15\times$ more than an error in the geometric factor, and it drives the gain from
$+0.099$ to $+0.018$. Fifth, gating becomes attractive only in the regime where
the registration candidate is already useless, with a gain bounded by $+0.017$.

Two methodological observations follow. The first concerns fusion-quality
evaluation: reference-based metrics computed against the observed (misaligned)
inputs reward leaving the misalignment uncorrected, so a study that reports only
these scores can conclude the opposite of the detection result. The second
concerns necessity supervision: a necessity map that correlates poorly with the
displacement can still be a good gate (the proxy map reaches $0.6792$), and a map
that correlates almost perfectly can be a mediocre gate (the factorised map
reaches $0.6709$). Correlation with geometry is therefore not a proxy for
downstream utility.

For practice, the results suggest a simple policy. Estimate the global
misalignment; if it is below a few pixels, do not correct; otherwise correct at
full strength with the best available registration, and invest the remaining
effort in registration quality rather than in spatial gating. Gating should be
reserved for systems in which the registration candidate is known to fail
locally, and it should then be driven by a reliability signal rather than by a
necessity map.

The limitations follow from the protocol. The perturbation is a global affine
transform, so the results do not establish what happens under depth-dependent
parallax, local non-rigid distortion, or occlusion; the real registrar is one
representative implementation of classic cross-modal registration, and a learned
matcher might behave differently; the error sweep assumes a spatially smooth
error field uncorrelated with image content, which is favourable to a
reliability-based gate; and the LLVIP experiment uses a single seed. Finally, the
detector is frozen and was trained on fused images from this project, so all
detection numbers describe zero-adaptation compatibility rather than the maximum
performance obtainable after fine-tuning."""

CONCLUSION = r"""This paper asked where a registration correction should be allowed to act,
and answered with a measured protocol rather than a new gate. Under controlled
affine misalignment, dense correction of the registered candidate is the best
policy, a zero-parameter displacement threshold saturates what a learned
necessity map can achieve, and the multiplicative factorisation of geometric
demand and task importance is significantly worse than the task map alone. With a
real registration module, warp error dominates geometric-factor error by
$15\times$ and gating becomes indistinguishable from dense correction; gating only
pays off once the mean registration error exceeds $\approx12$\,px, where the
correction itself is worthless. Taken together, registration necessity behaves as
a \emph{magnitude} decision: whether to correct at all, rather than where. We
release the protocol, the gate baselines and the per-image predictions so that
this conclusion, and the claims of future necessity-guided methods, can be
checked directly."""

# ---------------------------------------------------------------- surgery
out = src

# title
_ts = out.index("\\Title{")
_te = out.index("}\n", _ts) + 1
assert _ts < _te, "title block not found"
out = out[:_ts] + TITLE + out[_te:]

# abstract
m = re.search(r"\\abstract\{.*?\n\n\\keyword\{.*?\}", out, flags=re.S)
assert m, "abstract block not found"
out = out[:m.start()] + ABSTRACT + "\n\n" + KEYWORDS + out[m.end():]

# contributions
m = re.search(r"The contributions of this paper are summarized as follows\..*?\\end\{itemize\}",
              out, flags=re.S)
assert m, "contribution block not found"
out = out[:m.start()] + CONTRIB + out[m.end():]

# results (from 4.3 up to the Discussion section)
start = out.index(r"\subsection{Gated Correction Results and Statistical Analysis}")
end = out.index("%=================================================================\n\\section{Discussion}")
out = out[:start] + RESULTS.lstrip("\n") + "\n" + out[end:]

# discussion + conclusions
dstart = out.index(r"\section{Discussion}")
dend = out.index(r"\vspace{6pt}", dstart)
new_tail = ("\\section{Discussion}\n\\label{sec:discussion}\n\n" + DISCUSSION +
            "\n\n%=================================================================\n"
            "\\section{Conclusions}\n\\label{sec:conclusion}\n\n" + CONCLUSION + "\n\n")
out = out[:dstart] + new_tail + out[dend:]

# fix the diagnostic-figure text (the figure is a distribution plot, not a binned curve)
old_fig = re.search(r"Figure~\\ref\{fig:validity_binned\} shows the mean predicted necessity as a\nfunction.*?over its full extent\.", out, flags=re.S)
if old_fig:
    out = out[:old_fig.start()] + (
        "Figure~\\ref{fig:validity_binned} reports the distribution, over image\n"
        "pairs, of the association between each map and the true displacement. The\n"
        "proxy-supervised map is centred on chance, both measured-target maps are\n"
        "shifted to the right, and the factorised variant is tightly concentrated\n"
        "near one---as expected, because its geometric factor is supplied\n"
        "explicitly rather than inferred from appearance.") + out[old_fig.end():]

# remove the overreach sentence about earlier studies
out = out.replace(
    "This finding explains\nwhy proxy-supervised selective registration failed to influence downstream\ndetection in earlier studies.",
    "This finding is consistent with the\nobservation in~\\cite{li2024registration} that such maps are dominated by\nappearance; it does not by itself explain the behaviour of any earlier method,\nbecause we reproduce the proxy formulation and not their experiments.")
out = out.replace(
    "This finding explains why proxy-supervised selective registration failed to\ninfluence downstream detection in earlier studies.",
    "This finding is consistent with the observation in~\\cite{li2024registration}\nthat such maps are dominated by appearance; it does not by itself explain the\nbehaviour of any earlier method, because we reproduce the proxy formulation and\nnot their experiments.")

# replace the remaining references to the removed gamma figure with a table
out = out.replace("Figure~\\ref{fig:gamma} reports the complete sweep",
                  "Table~\\ref{tab:gamma} reports the sweep")
out = out.replace("Figure~\\ref{fig:gamma}, the main experiments use",
                  "Table~\\ref{tab:gamma}, the main experiments use")


# ---- post-processing: extra labels, gamma table, cost subsection -------------
COST = r"""
\subsection{Computational cost}
\label{sec:cost}

The necessity estimator contains $185\,345$ parameters, requires $4.69$\,GFLOPs per
$480\times640$ image, runs in $1.24$\,ms on an A100 (batch size one) with a
$34$\,MB peak footprint, and the bounded correction itself costs $0.027$\,ms.
This is the same order as the frozen detector's inference time ($\approx1.3$\,ms
per image), so the estimator is not negligible: in a deployment that already runs
a detector it roughly doubles the per-frame neural cost and must be justified by
accuracy rather than by speed. The registration module that supplies the candidate
is excluded, as it is a separate component.

"""
_disc_marker = "%=================================================================\n\\section{Discussion}"
if _disc_marker in out:
    out = out.replace(_disc_marker, COST + _disc_marker, 1)
out = out.replace("\\label{sec:gates}", "\\label{sec:gates}\n\\label{sec:stats}", 1)
out = out.replace("\\label{sec:magnitude}",
                  "\\label{sec:magnitude}\n\\label{sec:sweep}\n\\label{sec:crossdetector}\n\\label{sec:limits_detector}", 1)
out = out.replace("\\label{sec:discussion}", "\\label{sec:discussion}\n\\label{sec:limitations}", 1)
out = out.replace("\\subsection{The factorisation does not hold}",
                  "\\input{tables/tab_gamma}\n\n\\subsection{The factorisation does not hold}", 1)


EXTRA_CONDS = r"""
Beyond the conditions used in the original protocol, the revision evaluates seven
further gates so that every claim can be attributed to a specific design choice:
(i) a \emph{task-only} gate $N=\hat S_{\mathrm{task}}$, which removes the
geometric factor entirely; (ii) a \emph{proxy} gate built from the appearance
proxy of earlier work~\cite{li2024registration}, both analytically and from the
recovered proxy-supervised model; (iii) a zero-parameter \emph{displacement
threshold} $N=\mathbb{1}[\bar d>\tau]$ and (iv) a power rule $N=\bar d^{\alpha}$;
(v) a \emph{detector-box} rule that replaces the annotation map by the frozen
detector's own predictions; (vi) an \emph{absolute-scale} variant without
per-image min--max normalisation; and (vii) the six estimated-registration
conditions C0--C4 of Section~\ref{sec:estimated}. All of them use the same
registered candidate family, the same detector and the same evaluation pairs.

"""
_anchor = "\\item \\textbf{Uniform / shuffled / random}: controls that replace the spatial\nallocation while preserving the amount of correction.\n\\end{itemize}"
if _anchor in out:
    out = out.replace(_anchor, _anchor + EXTRA_CONDS, 1)
else:
    print("WARNING: conditions anchor not found")


CONSISTENCY = [
 ("shows empirically that the proxy is essentially uncorrelated with $d(x)$: it\nprimarily responds to appearance structure rather than to the amount of\ngeometric correction required.",
  "shows empirically that the proxy is essentially uncorrelated with $d(x)$: it\nprimarily responds to appearance structure rather than to the amount of\ngeometric correction required. As reported in Section~\\ref{sec:gates}, this\ndoes not make it a bad gate: used inside the same bounded correction, the\nproxy-supervised map reaches $0.6792$ and therefore outperforms the learned\nfactorised gate. Correlation with geometric demand and downstream utility are\ndifferent properties, and we report both."),
 ("Because the estimator is small and the\ncorrection is a convex combination, the method is compatible with online\noperation; the measured latency and memory footprint are reported in\nSection~\\ref{sec:cost}.",
  "The estimator runs in $1.24$\\,ms per frame, which is the same order as the\nfrozen detector; it is therefore compatible with online operation, but only if\nits accuracy justifies roughly doubling the per-frame neural cost\n(Section~\\ref{sec:cost})."),
]
for old, new in CONSISTENCY:
    if old in out:
        out = out.replace(old, new, 1)
    else:
        print("WARNING: consistency anchor not found:", old[:60])


POSITIONING = [
 ("\\caption{Positioning of representative strategies for misaligned\ninfrared--visible fusion. Among the representative strategies listed here, the present work is the one\nwhose spatial decision is supervised by a measurable target that explicitly\ncontains registration demand and task importance.}",
  "\\caption{Positioning of representative strategies for misaligned\ninfrared--visible fusion. The present work does not introduce another gate; it\nsupplies the measurable protocol and the controlled baselines needed to test\nwhether a spatial decision is justified at all, and reports the outcome of that\ntest.}"),
 ("\\textbf{This work} & IR--VIS & bounded correction over pixel average & learned\nfrom a measurable target & \\textbf{ground-truth displacement and downstream\ndetection} \\\\",
  "\\textbf{This work} & IR--VIS & correction of a registered candidate; gating\nstudied as a design choice & twelve gate maps, from zero-parameter rules to\nlearned maps & \\textbf{exact displacement, real registration error, and\ndownstream detection} \\\\"),
]
for old, new in POSITIONING:
    if old in out:
        out = out.replace(old, new, 1)
    else:
        print("WARNING: positioning anchor not found")


DATA_AVAIL = (
 "\\dataavailability{The M3FD and LLVIP datasets are publicly available (see\n"
 "references). An anonymised repository accompanies this submission and contains\n"
 "the experiment code, the six trained checkpoints plus the recovered proxy model,\n"
 "the evaluation manifests, the controlled-misalignment parameters, the per-image\n"
 "predictions, the fusion-metric implementation, the table builders and a\n"
 "single-command reproduction file; SHA256 checksums are included and every number\n"
 "in the paper is indexed in a results ledger. We intend to replace the anonymised\n"
 "link with a permanent DOI upon acceptance.}\n")
_av = out.index("\\dataavailability{")
_ave = out.index("}\n", out.index("reasonable request.", _av)) + 2
if _av < _ave:
    out = out[:_av] + DATA_AVAIL + out[_ave:]
else:
    print("WARNING: data availability block not replaced")


# ---- detector provenance, phi/s_g table, metadata, author contributions -------
PROVENANCE = r"""
\paragraph{Downstream detector provenance.}
The detector used for every reported number is a YOLOv8n fine-tuned for the M3FD
label space. Its configuration was recovered from the checkpoint metadata and
cross-checked against the data configuration used for training: initialisation
from the official YOLOv8n weights, 50 epochs at $640$\,px with batch size $16$,
AdamW with an initial learning rate of $10^{-3}$, weight decay $5\times10^{-4}$,
mosaic augmentation and \texttt{close\_mosaic}$=10$, a fixed seed and
deterministic training. The training inputs are the grayscale fused images of the
3151 training pairs produced by this project, and the checkpoint was selected by
validation mAP on the M3FD infrared test split. Two consequences are disclosed
rather than hidden: the selection uses the split that is later evaluated, and the
training inputs are fused outputs of this project, which favours pixel-average-like
inputs in the comparison of Section~\ref{sec:fair}. For the cross-detector check we
additionally use an independently trained YOLOv8n on M3FD pixel-average images,
and for LLVIP the documented RT-DETR-L detector; both weights, their hashes and
their training configurations are released with the anonymised repository.

"""
_impl = "\\subsubsection{Detection Protocol}"
if _impl in out:
    out = out.replace(_impl, PROVENANCE + _impl, 1)
else:
    print("WARNING: detection protocol anchor not found")

_phi_anchor = "\\input{tables/tab_gamma}"
if _phi_anchor in out:
    out = out.replace(_phi_anchor, _phi_anchor + "\n\\input{tables/tab_phi_sg}", 1)
else:
    print("WARNING: gamma anchor not found")

_phi_text = (
 "Table~\\ref{tab:phisg} reports the sensitivity of the two free constants on the\n"
 "held-out validation subset. $\\phi$ changes the factorised gate by at most $0.010$\n"
 "over its full range, with the best value at the default $0.5$; $s_g$ matters more,\n"
 "and its best value is $8$\\,px rather than the default $16$\\,px---another instance\n"
 "of the strength effect, since a smaller saturation scale increases the effective\n"
 "correction. Neither constant was tuned on evaluation data, and we do not claim\n"
 "that the defaults are globally optimal.\n\n")
_anchor2 = "\\subsection{The factorisation does not hold}"
if _anchor2 in out:
    out = out.replace(_anchor2, _phi_text + _anchor2, 1)

# NOTE: the MDPI class writes /Subject from the abstract and overrides later
# hypersetup calls; we therefore leave the class behaviour in place and
# note in STATUS.md that /Subject mirrors the abstract (math symbols lost).

# author contributions: use full names (Z.Z. is ambiguous between two authors)
_contrib_old = ("\\authorcontributions{Conceptualization, R.W. and Z.Z.; methodology, R.W.;\n"
                "software, R.W. and Z.Z.; validation, all authors; writing---original draft\n"
                "preparation, R.W.; writing---review and editing, all authors. All authors have\n"
                "read and agreed to the published version of the manuscript.}")
_contrib_new = ("\\authorcontributions{Conceptualization, Rongchao Wang and Zhaofa Zhou;\n"
                "methodology, Rongchao Wang; software, Rongchao Wang and Zhaofa Zhou;\n"
                "validation, all authors; formal analysis, Rongchao Wang; writing---original\n"
                "draft preparation, Rongchao Wang; writing---review and editing, Zhaofa Zhou\n"
                "and Zhili Zhang; supervision, Zhili Zhang. All authors have read and agreed\n"
                "to the published version of the manuscript. \\textbf{[Authors: please confirm\n"
                "the individual contributions; the previous ``Z.Z.'' abbreviation was\n"
                "ambiguous between Zhaofa Zhou and Zhili Zhang and has been expanded as\n"
                "stated here.]}}")
if _contrib_old in out:
    out = out.replace(_contrib_old, _contrib_new, 1)
else:
    print("WARNING: author contributions anchor not found")

DST.write_text(out)
print("wrote", DST, len(out.splitlines()), "lines")
print("contains old tables:", [k for k in ("tab:main", "tab:detcount", "tab:bootstrap", "tab:published") if "\\ref{%s}" % k in out])
