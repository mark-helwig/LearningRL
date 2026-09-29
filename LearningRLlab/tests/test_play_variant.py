"""AST scan: every registered task must have a ``-PLAY`` twin that plays by the rule.

See ``CLAUDE.md``'s "the ``-PLAY`` rule". Checks all three parts: every
``gym.register`` id has a ``<id>-PLAY`` sibling; that sibling points at its own env
cfg class, not the training one; that class sets ``scene.num_envs = 100`` and
re-anchors the viewer to an asset.

AST scanning, not importing: ``import LearningRLlab.tasks`` pulls in ``omni.*``, so an
import-based test could only run inside the container. Parsing the source keeps
this in the ``static`` gate, which runs on Windows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "LearningRLlab" / "tasks"

PLAY_SUFFIX = "-PLAY"
REQUIRED_NUM_ENVS = 100


def _literal_prefix(node: ast.expr) -> str | None:
    """Return the literal text of a str constant, or of an f-string's constant parts.

    Entry points are written as ``f"{__name__}.mod:Cls"``, whose only non-literal piece
    is ``__name__``. Dropping the interpolations leaves ``".mod:Cls"`` -- enough to
    resolve the module file relative to the registering package and to name the class.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(p.value for p in node.values if isinstance(p, ast.Constant) and isinstance(p.value, str))
    return None


def _registrations() -> dict[str, tuple[Path, str | None]]:
    """Map every ``gym.register`` id to the file it lives in and its env cfg entry point."""
    found: dict[str, tuple[Path, str | None]] = {}
    for init_py in sorted(TASKS_ROOT.rglob("__init__.py")):
        tree = ast.parse(init_py.read_text(encoding="utf-8"), filename=str(init_py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "register"):
                continue

            task_id = None
            entry_point = None
            for kw in node.keywords:
                if kw.arg == "id":
                    task_id = _literal_prefix(kw.value)
                elif kw.arg == "kwargs" and isinstance(kw.value, ast.Dict):
                    for key, value in zip(kw.value.keys, kw.value.values):
                        if isinstance(key, ast.Constant) and key.value == "env_cfg_entry_point":
                            entry_point = _literal_prefix(value)
            if task_id is not None:
                found[task_id] = (init_py, entry_point)
    return found


def _cfg_class(init_py: Path, entry_point: str) -> ast.ClassDef:
    """Resolve a ``"<pkg>.<module>:<Class>"`` entry point to its ``ClassDef`` node."""
    module_part, _, class_name = entry_point.partition(":")
    module_file = init_py.parent / f"{module_part.rsplit('.', 1)[-1]}.py"
    assert module_file.is_file(), f"{entry_point} names {module_file}, which does not exist"

    tree = ast.parse(module_file.read_text(encoding="utf-8"), filename=str(module_file))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    pytest.fail(f"{entry_point} names class {class_name}, which is not defined in {module_file}")


def _assigned_attrs(class_node: ast.ClassDef) -> dict[str, ast.expr]:
    """Collect ``self.a.b = value`` assignments anywhere in a class body, keyed ``"a.b"``."""
    assigned: dict[str, ast.expr] = {}
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            parts: list[str] = []
            cursor: ast.expr = target
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name) and cursor.id == "self" and parts:
                assigned[".".join(reversed(parts))] = node.value
    return assigned


@pytest.mark.static
def test_scan_finds_registrations():
    """Guard the scan itself: a silently-empty walk would make every test below vacuous."""
    assert _registrations(), f"no gym.register(...) calls found under {TASKS_ROOT} -- the AST scan is broken"


@pytest.mark.static
def test_every_task_has_a_play_variant():
    registrations = _registrations()
    missing = [
        task_id
        for task_id in registrations
        if not task_id.endswith(PLAY_SUFFIX) and f"{task_id}{PLAY_SUFFIX}" not in registrations
    ]
    assert not missing, (
        f"tasks registered with no {PLAY_SUFFIX} twin: {missing}. Every task needs one -- "
        'see CLAUDE.md, "the -PLAY rule".'
    )


@pytest.mark.static
def test_play_variants_use_their_own_env_cfg():
    registrations = _registrations()
    for task_id, (_, entry_point) in registrations.items():
        if not task_id.endswith(PLAY_SUFFIX):
            continue
        base_id = task_id[: -len(PLAY_SUFFIX)]
        base = registrations.get(base_id)
        assert entry_point is not None, f"{task_id} registers no env_cfg_entry_point"
        assert base is None or entry_point != base[1], (
            f"{task_id} reuses {base_id}'s env cfg ({entry_point}). A -PLAY task needs its own "
            "cfg class, or it is just a second name for the training scene."
        )


@pytest.mark.static
def test_play_cfgs_set_num_envs_and_follow_a_robot():
    for task_id, (init_py, entry_point) in _registrations().items():
        if not task_id.endswith(PLAY_SUFFIX):
            continue
        assigned = _assigned_attrs(_cfg_class(init_py, entry_point))

        num_envs = assigned.get("scene.num_envs")
        assert isinstance(num_envs, ast.Constant) and num_envs.value == REQUIRED_NUM_ENVS, (
            f"{task_id}'s cfg must set `self.scene.num_envs = {REQUIRED_NUM_ENVS}` (CLAUDE.md, the -PLAY rule)"
        )

        origin_type = assigned.get("viewer.origin_type")
        assert isinstance(origin_type, ast.Constant) and origin_type.value in {"asset_root", "asset_body"}, (
            f"{task_id}'s cfg must anchor the viewer to a robot -- set `self.viewer.origin_type` to "
            '"asset_root" (or "asset_body"), not leave it at the world-anchored default'
        )
        assert "viewer.asset_name" in assigned, f"{task_id}'s cfg sets viewer.origin_type but no viewer.asset_name"
