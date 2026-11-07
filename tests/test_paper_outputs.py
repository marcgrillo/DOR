"""Paper tables and figures rebuilt from the committed results must match the committed outputs.

The scripts in scripts/paper/ are imported as modules, their module-level ``OUT`` is pointed
at a temporary directory and their ``main()`` is called:

* Table 3 (outputs/tables/summary_table_v1.tex) and Supplementary Tables B.1-B.9
  (outputs/tables/appendix_tables_v1.tex), from results/grid_results_joint_v1.npz, compared
  as text with line endings normalised (a checkout with CRLF endings compares equal);
* Figures 2-4 (outputs/figures/fig_factors_bdor_v1.png, fig_baselines_t1_v1.png,
  fig_sets_tau_v1.png), from the npz and results/sets_tau_comparison.csv, compared byte for
  byte and, if the bytes differ, pixel for pixel. Rendering depends on the matplotlib version
  and the fonts, so a difference fails the test only in the reference environment (the
  matplotlib pinned in requirements.txt, on Windows) and is reported as a skip elsewhere.

Figure 5 needs the dataset folders under data/ and is not covered here.
"""
import difflib
import importlib.util
import sys

import numpy as np
import pytest

from bdor.paths import FIGURES_DIR, REPO_ROOT, TABLES_DIR

PAPER_SCRIPTS = REPO_ROOT / "scripts" / "paper"


def _load_script(name, monkeypatch):
    """Import scripts/paper/<name>.py as a module (its folder on sys.path, as when run)."""
    monkeypatch.syspath_prepend(str(PAPER_SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"paper_script_{name}",
                                                  PAPER_SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _redirect(module, target):
    """Point the script's ``OUT`` at ``target``, keeping the type (str or Path) it uses."""
    module.OUT = str(target) if isinstance(module.OUT, str) else target


def _read_text(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


# --------------------------------------------------------------------------------- tables

@pytest.mark.parametrize("script, table", [
    ("make_summary_table", "summary_table_v1.tex"),        # Table 3
    ("make_appendix_tables", "appendix_tables_v1.tex"),    # Tables B.1-B.9
])
def test_table_regenerates_identically(script, table, tmp_path, monkeypatch, capsys):
    module = _load_script(script, monkeypatch)
    _redirect(module, tmp_path / table)
    module.main()
    capsys.readouterr()                                     # drop the script's report

    new, committed = _read_text(tmp_path / table), _read_text(TABLES_DIR / table)
    if new != committed:
        diff = difflib.unified_diff(committed.splitlines(), new.splitlines(),
                                    f"outputs/tables/{table}", "regenerated", lineterm="", n=0)
        pytest.fail("regenerated table differs from the committed one:\n"
                    + "\n".join(list(diff)[:40]))


# -------------------------------------------------------------------------------- figures

FIGURES = ["fig_factors_bdor_v1.png",    # Figure 2
           "fig_baselines_t1_v1.png",    # Figure 3
           "fig_sets_tau_v1.png"]        # Figure 4


@pytest.fixture(scope="module")
def regenerated_figures(tmp_path_factory):
    out = tmp_path_factory.mktemp("figures")
    with pytest.MonkeyPatch.context() as mp:
        module = _load_script("make_figures", mp)
        _redirect(module, out)
        module.main(["--only", "fig2", "fig3", "fig4"])
    return out


def _pinned_version(package):
    """Version pinned for ``package`` in requirements.txt (None if not pinned)."""
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        name, sep, version = line.split("#")[0].partition("==")
        if sep and name.strip().lower() == package:
            return version.strip()
    return None


@pytest.mark.parametrize("name", FIGURES)
def test_figure_regenerates_identically(regenerated_figures, name):
    new, committed = regenerated_figures / name, FIGURES_DIR / name
    if new.read_bytes() == committed.read_bytes():
        return
    # the PNG compressor alone can change the bytes: compare the decoded pixels
    import matplotlib
    import matplotlib.image as mpimg
    a, b = mpimg.imread(new), mpimg.imread(committed)
    if a.shape == b.shape and np.array_equal(a, b):
        return
    pinned = _pinned_version("matplotlib")
    if matplotlib.__version__ != pinned or sys.platform != "win32":
        pytest.skip(f"{name} differs from the committed image under matplotlib "
                    f"{matplotlib.__version__} on {sys.platform}; the exact image check "
                    f"needs the reference environment (matplotlib {pinned}, Windows)")
    pytest.fail(f"{name}: regenerated image differs from the committed one "
                f"(pixel arrays {a.shape} vs {b.shape})")
