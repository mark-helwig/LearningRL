"""AST scan: LearningRLlab source must never name a Unitree robot cfg from ``isaaclab_assets``.

AST source scanning, not an import hook: a task that subclasses an upstream env cfg
(e.g. ``G1FlatEnvCfg`` -> ``G1RoughEnvCfg``) transitively imports
``isaaclab_assets.G1_MINIMAL_CFG``, so an import hook would false-positive on
upstream's line, not ours. AST scanning matches on the **symbol**, not the module
(``CARTPOLE_CFG`` passes, ``G1_CFG`` fails), runs in milliseconds with no Isaac Lab
installed, so it works on Windows too.

Deliberate limitation: cannot catch string-built access
(``getattr(isaaclab_assets, "G1_" + "CFG")``) -- defends against an absent-minded
autocomplete import, not an adversary hiding a violation.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["LearningRLlab", "scripts"]

# Unitree configs shipped by isaaclab_assets. LearningRLlab robot cfgs must come from
# robot_description instead (imported directly by each task's env cfg).
BANNED_SYMBOLS = {
    "G1_CFG",
    "G1_MINIMAL_CFG",
    "G1_29DOF_CFG",
    "G1_INSPIRE_FTP_CFG",
    "H1_CFG",
    "H1_MINIMAL_CFG",
    "UNITREE_A1_CFG",
    "UNITREE_GO1_CFG",
    "UNITREE_GO2_CFG",
    "GO1_ACTUATOR_CFG",
}

BANNED_MODULE_ROOT = "isaaclab_assets"
BANNED_SUBMODULE = "isaaclab_assets.robots.unitree"


@dataclass(frozen=True)
class Violation:
    """One offending reference found by the scan."""

    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _is_banned_root_module(module: str | None) -> bool:
    if module is None:
        return False
    return module == BANNED_MODULE_ROOT or module.startswith(BANNED_MODULE_ROOT + ".")


def _is_banned_submodule(module: str | None) -> bool:
    if module is None:
        return False
    return module == BANNED_SUBMODULE or module.startswith(BANNED_SUBMODULE + ".")


def scan_source(source: str, path: Path) -> list[Violation]:
    """Return every banned isaaclab_assets Unitree reference found in ``source``.

    ``path`` is used only to label violations -- ``source`` need not actually live at
    ``path`` (the planted-violation test below scans in-memory strings against
    throwaway paths).
    """
    violations: list[Violation] = []
    tree = ast.parse(source, filename=str(path))

    # Names in this module bound to the isaaclab_assets module itself (via
    # `import isaaclab_assets` or `import isaaclab_assets as ia`), so attribute access
    # like `isaaclab_assets.G1_CFG` / `ia.G1_CFG` can also be caught.
    module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == BANNED_MODULE_ROOT:
                    module_aliases.add(alias.asname or alias.name)
                elif _is_banned_submodule(alias.name):
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            f"imports '{alias.name}' (isaaclab_assets' Unitree robot module); "
                            "LearningRLlab robot cfgs must come from robot_description instead.",
                        )
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import: cannot possibly be isaaclab_assets

            module = node.module
            if _is_banned_submodule(module):
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        f"imports from '{module}' (isaaclab_assets' Unitree robot module); "
                        "LearningRLlab robot cfgs must come from robot_description instead.",
                    )
                )
                continue

            if _is_banned_root_module(module):
                for alias in node.names:
                    if alias.name == "*":
                        violations.append(
                            Violation(
                                path,
                                node.lineno,
                                f"wildcard import from '{module}' -- this can silently pull in a "
                                "Unitree cfg. Import specific, non-Unitree symbols instead.",
                            )
                        )
                    elif alias.name in BANNED_SYMBOLS:
                        violations.append(
                            Violation(
                                path,
                                node.lineno,
                                f"imports banned symbol '{alias.name}' from '{module}'. LearningRLlab "
                                "robot cfgs must come from robot_description, not isaaclab_assets "
                                "(import from robot_description.robots.<robot> instead).",
                            )
                        )
                    else:
                        # A non-banned name imported FROM isaaclab_assets is still a foothold:
                        #     from isaaclab_assets import robots
                        #     robot = robots.unitree.G1_CFG
                        # binds `robots` locally, so track it as an alias and let the attribute
                        # walk below catch any banned symbol reached through it.
                        module_aliases.add(alias.asname or alias.name)

        elif isinstance(node, ast.Attribute):
            # Walk the FULL attribute chain back to its root Name before deciding.
            #
            # Checking only `node.value` (one level) misses the realistic case
            #     import isaaclab_assets
            #     robot = isaaclab_assets.robots.unitree.G1_CFG
            # where node.value is itself an Attribute, not a Name. That is ordinary
            # autocomplete behaviour, not adversarial obfuscation, so it must be caught.
            if node.attr in BANNED_SYMBOLS:
                root = node.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in module_aliases:
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            f"accesses '{ast.unparse(node)}' (isaaclab_assets); LearningRLlab robot "
                            "cfgs must come from robot_description instead.",
                        )
                    )

    return violations


def _scan_file(path: Path) -> list[Violation]:
    return scan_source(path.read_text(encoding="utf-8"), path)


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_DIRS:
        base = REPO_ROOT / rel
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    return files


@pytest.mark.static
def test_no_unitree_configs_from_isaaclab_assets():
    """The real check: no LearningRLlab source file may name a Unitree cfg from isaaclab_assets."""
    all_violations: list[Violation] = []
    for path in _iter_source_files():
        all_violations.extend(_scan_file(path))

    assert not all_violations, "Found forbidden isaaclab_assets Unitree references:\n" + "\n".join(
        str(v) for v in all_violations
    )


@pytest.mark.static
def test_scan_actually_covers_source():
    """Guard against the scan silently matching zero files after a layout move.

    Without this, ``test_no_unitree_configs_from_isaaclab_assets`` could pass forever
    for the wrong reason -- because ``SCAN_DIRS`` stopped matching anything -- and
    nobody would notice.
    """
    files = _iter_source_files()
    assert len(files) >= 5, f"Expected at least 5 source files under {SCAN_DIRS}, found {len(files)}: {files}"

    scanned_names = {p.name for p in files}
    assert "g1_env_cfg.py" in scanned_names, (
        "g1_env_cfg.py was not found by the scan -- SCAN_DIRS "
        f"({SCAN_DIRS}) is out of sync with the actual package layout."
    )


@pytest.mark.static
def test_detects_a_planted_violation(tmp_path):
    """A source-scanning test never shown to fail is indistinguishable from `assert True`.

    Plants every shape of violation the scanner claims to catch, plus legitimate uses it
    must NOT flag. The dotted-chain cases are regression tests: an earlier version walked
    only one attribute level, so `isaaclab_assets.robots.unitree.G1_CFG` -- ordinary
    autocomplete, not obfuscation -- passed silently.
    """
    direct_import_src = textwrap.dedent(
        """
        from isaaclab_assets import G1_CFG
        """
    )
    attribute_access_src = textwrap.dedent(
        """
        import isaaclab_assets

        robot = isaaclab_assets.G1_CFG
        """
    )
    legitimate_src = textwrap.dedent(
        """
        from isaaclab_assets import CARTPOLE_CFG
        """
    )

    assert scan_source(direct_import_src, tmp_path / "direct_violation.py"), (
        "direct banned import 'from isaaclab_assets import G1_CFG' was not detected"
    )
    assert scan_source(attribute_access_src, tmp_path / "attribute_violation.py"), (
        "attribute access 'isaaclab_assets.G1_CFG' was not detected"
    )
    assert not scan_source(legitimate_src, tmp_path / "legitimate.py"), (
        "legitimate 'from isaaclab_assets import CARTPOLE_CFG' was incorrectly flagged"
    )

    # --- regression: multi-level attribute chains -------------------------------
    dotted_chain_src = textwrap.dedent(
        """
        import isaaclab_assets

        robot = isaaclab_assets.robots.unitree.G1_CFG
        """
    )
    from_submodule_src = textwrap.dedent(
        """
        from isaaclab_assets import robots

        robot = robots.unitree.G1_CFG
        """
    )
    aliased_module_src = textwrap.dedent(
        """
        import isaaclab_assets as ia

        robot = ia.robots.unitree.H1_CFG
        """
    )
    legitimate_chain_src = textwrap.dedent(
        """
        import isaaclab_assets

        robot = isaaclab_assets.robots.anymal.ANYMAL_C_CFG
        """
    )

    assert scan_source(dotted_chain_src, tmp_path / "dotted.py"), (
        "dotted chain 'isaaclab_assets.robots.unitree.G1_CFG' was not detected"
    )
    assert scan_source(from_submodule_src, tmp_path / "from_submodule.py"), (
        "'from isaaclab_assets import robots' + 'robots.unitree.G1_CFG' was not detected"
    )
    assert scan_source(aliased_module_src, tmp_path / "aliased.py"), (
        "aliased module chain 'ia.robots.unitree.H1_CFG' was not detected"
    )
    assert not scan_source(legitimate_chain_src, tmp_path / "legitimate_chain.py"), (
        "legitimate 'isaaclab_assets.robots.anymal.ANYMAL_C_CFG' was incorrectly flagged"
    )
