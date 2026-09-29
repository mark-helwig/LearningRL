"""List Gymnasium environments registered by LearningRLlab.

Rewritten from upstream's ``scripts/environments/list_envs.py`` to drop
``AppLauncher``: gym registration needs no simulation app, so this script never
starts Kit and does not depend on whether ``AppLauncher(headless=True)`` behaves
correctly (which check 7 needs to be independent of).

Upstream's own ``list_envs.py`` filters on ``"Isaac" in spec.id`` and never imports
LearningRLlab, so it will never show ``LearningRL-G1-Walk-v0`` -- that is expected, not a bug in
either script.
"""

import argparse
import sys

import gymnasium as gym

import LearningRLlab.tasks  # noqa: F401  (registers the LearningRL tasks with gymnasium)

# Stdlib table rendering, not `prettytable` (upstream's choice) -- not a declared
# or image-installed dependency, so relying on it would fail with a
# ModuleNotFoundError that looks like "the extension did not load".


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a left-aligned fixed-width table. No third-party dependency."""
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i]) for i in range(len(headers))
    ]
    rule = "-+-".join("-" * w for w in widths)
    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, widths)), rule]
    lines.extend(" | ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def main() -> int:
    """Print a table of LearningRLlab-registered environments; return a process exit code."""
    parser = argparse.ArgumentParser(description="List Gymnasium environments registered by LearningRLlab.")
    parser.add_argument(
        "--filter",
        type=str,
        default="LearningRL",
        help="Only list environment ids containing this substring (default: 'LearningRL').",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print one bare environment id per line, for grep/scripting.",
    )
    args = parser.parse_args()

    rows = []
    for env_id, spec in sorted(gym.registry.items()):
        if args.filter not in env_id:
            continue
        kwargs = spec.kwargs or {}
        rows.append(
            [
                env_id,
                str(spec.entry_point),
                str(kwargs.get("env_cfg_entry_point", "-")),
                str(kwargs.get("rsl_rl_cfg_entry_point", "-")),
            ]
        )

    if args.plain:
        for row in rows:
            print(row[0])
    else:
        print(_render_table(["id", "entry_point", "env cfg", "agent cfg"], rows))

    if not rows:
        # Nonzero exit so REPO 1's 50-verify.sh can script "did LearningRLlab actually
        # register anything" as a single command instead of parsing table output.
        print(f"No environments matched filter '{args.filter}'.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
