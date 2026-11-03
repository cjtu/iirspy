import ast
from pathlib import Path

ROOT = Path(__file__).parents[1] / "iirspy"
BASE = {"config.py", "instrument.py", "io.py", "workspace.py"}
LEVELS = {"l1", "l2", "georef", "cli"}
FORBIDDEN = {"l1": {"l2", "georef", "cli"}, "georef": {"l2", "cli"}, "l2": {"cli"}}


def _imported_packages(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [f"{node.module}.{a.name}" for a in node.names] if node.module == "iirspy" else [node.module or ""]
        else:
            continue
        for n in names:
            parts = n.split(".")
            if parts[0] == "iirspy" and len(parts) > 1:
                yield parts[1]


def test_imports_point_down_the_layers():
    bad = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        layer = rel.parts[0] if len(rel.parts) > 1 else rel.name
        banned = FORBIDDEN.get(layer, set()) | (LEVELS if layer in BASE else set()) | ({"cli"} - {layer})
        bad += [f"{rel} imports iirspy.{p}" for p in set(_imported_packages(path)) & banned]
    assert not bad, "\n".join(bad)
