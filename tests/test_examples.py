"""The interactive elicitation example (examples/interactive_elicitation/inference_tool.py) runs
on its shipped data.json: one FTRL-DOR fit, one BAYES-DOR chain and one plot per step.

The example writes its plots next to the JSON file it reads, so it runs on a copy in a
temporary directory and the repository stays untouched.
"""
import importlib.util
import json
import shutil

from bdor.paths import REPO_ROOT

EXAMPLE_DIR = REPO_ROOT / "examples" / "interactive_elicitation"


def test_inference_tool_on_the_shipped_data(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data.json"
    shutil.copy(EXAMPLE_DIR / "data.json", data)
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    spec = importlib.util.spec_from_file_location("inference_tool_example",
                                                  EXAMPLE_DIR / "inference_tool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.run_inference_tool(str(data))
    capsys.readouterr()

    n_steps = len(json.loads(data.read_text())["steps"])
    assert n_steps >= 1
    written = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".png")
    assert written == sorted(f"utilities_step_{k}.png" for k in range(1, n_steps + 1))
    assert all((tmp_path / name).stat().st_size > 0 for name in written)
