"""Title-less paper versions of the figures used in NeurIPS2026-Workshop/main.tex.

Re-runs the owning plot scripts unmodified, but with every figure-level title
suppressed (the title moves into the LaTeX caption) and output redirected into
this folder. Per-panel axis titles (benign-dataset names, vendor names) are
kept — only the big suptitle + dataset-annotation block is dropped.

Run:  python make_paper_figures.py
"""

import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

BASE = Path(__file__).resolve().parent            # refined-figures/paper/
RF = BASE.parent                                  # refined-figures/
REPO = RF.parent
sys.path.insert(0, str(RF))
import style  # noqa: E402

# 1) No-op the shared suptitle + dataset-line helper (all refined scripts).
style.title_and_dataset = lambda *a, **k: None
# 2) No-op direct fig.suptitle calls (plot_by_vendor.py's two-line title).
Figure.suptitle = lambda self, *a, **k: None
# 3) Crop the now-empty title band even for scripts that don't use style.apply().
matplotlib.rcParams["savefig.bbox"] = "tight"

# 4) Redirect every savefig into this folder (basename only).
# For the two side-by-side sysdef panels (main-text Fig 3) inject an axes-level
# panel title — they sit in adjacent minipages, so each needs its own heading.
_orig_savefig = Figure.savefig

PANEL_TITLES = {
    "sysdef_asr.png": "Arena IPI Attack Success Rate",
    "sysdef_utility.png": "AgentDojo Utility (null injection)",
}


def _savefig(self, fname, *a, **k):
    name = Path(fname).name
    if name in PANEL_TITLES:
        self.axes[0].set_title(PANEL_TITLES[name], fontsize=19, fontweight="bold", pad=14)
    return _orig_savefig(self, BASE / name, *a, **k)


Figure.savefig = _savefig

SCRIPTS = [
    # plot_classifier_roc.py runs below via importlib (paper label patches).
    RF / "plot_system_defenses.py",       # sysdef_asr, sysdef_utility, sysdef_class_heatmap
    RF / "plot_attacks_vs_breaks.py",     # attacks_vs_breaks_users
    RF / "plot_effort_frontier.py",       # effort_frontier (needs user_attack_analysis/scan/*.tsv.gz)
    RF / "plot_next_break_cost.py",       # next_break_cost
    REPO / "strategy_frontier_panel" / "src" / "plot_by_vendor.py",  # asr_by_vendor (+ json side-car)
    BASE / "plot_asr_by_vendor_paper.py",  # asr_by_vendor_2x2 (reads the json written above)
    BASE / "plot_open_frontier_transfer_paper.py",  # open_frontier_transfer (OpenAI|Gemini 2-panel)
    BASE / "plot_open_frontier_transfer_anthropic_paper.py",  # anthropic appendix figure
]

for script in SCRIPTS:
    if not script.exists():  # payload-bearing builders are excluded from the public release
        print(f"-- skip (missing) {script.relative_to(REPO)}")
        continue
    print(f"== {script.relative_to(REPO)}")
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")

# ROC figures via importlib so paper-only label patches reach the functions
# (runpy.run_path returns a COPY of the namespace, so mutating its dict never
# reaches them). Anonymity: label "Cygnal" without the model size everywhere.
print("== plot_classifier_roc.py (full appendix versions)")
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("roc_mod", RF / "plot_classifier_roc.py")
_roc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_roc)
_roc.DEF_LABEL["cygnal-code-8b"] = "Cygnal"  # don't reveal model size
_roc.main()  # roc_ipi_head.png, roc_policy_head.png

# Reduced main-text variant: additionally drop the Gemini-3-Flash + Qwen3.5-9B
# judges and the unreliable Granite-think curve, and no highlighted star for
# Cygnal (anonymity).
print("== plot_classifier_roc.py (reduced main-text variant)")
_roc.ORDER = [d for d in _roc.ORDER
              if d not in ("llmjudge-gemini-3-flash", "llmjudge-qwen3.5-9b",
                           "granite-guardian-4.1-8b-think")]
_roc.CYGNAL = None  # nothing matches -> no star anywhere (panels + legend)
_roc.fig_head("ipi", "roc_ipi_head_main.png")
_roc.fig_head("policy", "roc_policy_head_main.png")
_roc.fig_combined("roc_combined_main.png")  # main text: both heads on one figure

# Strategy-survival figures (main-text, right after the ROC figure; + appendix set)
# via importlib so the anonymity label patch reaches the module functions.
print("== plot_strategy_survival.py (main + appendix)")
_spec_ss = importlib.util.spec_from_file_location("ss_mod", RF / "plot_strategy_survival.py")
_ss = importlib.util.module_from_spec(_spec_ss)
_spec_ss.loader.exec_module(_ss)
_ss.LABEL["cygnal-code-8b"] = "Cygnal"  # don't reveal model size
_ss.main()

# Strategy-robustness figure (main-text §trends) via importlib for the same
# reason. Anonymity: target-model names (incl. unreleased eval models) are
# dropped — the x-axis becomes rank 1..N, most-robust to least, and the
# per-model ASR/n annotations go with them.
print("== plot_strategy_robust.py (paper: rank-anonymized x-axis)")
_spec_sr = importlib.util.spec_from_file_location("srh_mod", RF / "plot_strategy_robust.py")
_srh = importlib.util.module_from_spec(_spec_sr)
_spec_sr.loader.exec_module(_srh)
_orig_sr_setup = _srh.setup


def _rank_setup(ax, models, show_y):
    _orig_sr_setup(ax, models, show_y)
    ax.set_xticklabels([str(i + 1) for i in range(len(models))], rotation=0,
                       fontsize=9, color=_srh.INK2)
    ax.set_xlabel("target model, ranked most robust → least", fontsize=9,
                  color=_srh.INK2)


_srh.setup = _rank_setup
_srh.main()  # strategy_robust_holds.png

print(f"done -> {BASE}")
