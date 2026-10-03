"""Package metadata tests: version sync with pyproject.toml and import purity."""

import ast
import sys
import tomllib
from pathlib import Path

import mcp_dynamic_tool_registration

# The library may only import the standard library, its declared runtime
# dependencies and itself: no web framework, error tracker or host app.
ALLOWED_TOP_LEVEL = set(sys.stdlib_module_names) | {
    "__future__",
    "mcp",
    "pydantic",
    "mcp_dynamic_tool_registration",
}


def test_version_matches_pyproject():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == mcp_dynamic_tool_registration.__version__


def _forbidden_imports(source: str) -> list[str]:
    """Return the forbidden top-level module names imported by ``source``."""
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue
            names = [node.module] if node.module else []
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top not in ALLOWED_TOP_LEVEL:
                found.append(name)
    return found


def test_no_forbidden_imports():
    package_dir = (
        Path(__file__).resolve().parents[1] / "src" / "mcp_dynamic_tool_registration"
    )
    python_files = sorted(package_dir.rglob("*.py"))
    assert python_files, f"no python files found under {package_dir}"
    for path in python_files:
        violations = _forbidden_imports(path.read_text(encoding="utf-8"))
        assert not violations, f"{path} imports forbidden module(s): {violations}"
