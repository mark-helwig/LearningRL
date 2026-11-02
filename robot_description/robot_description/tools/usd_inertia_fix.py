"""Undo Isaac Sim's URDF importer rotating every link's inertia axes.

The importer diagonalises each link's URDF inertia tensor and writes the principal moments
(`physics:diagonalInertia`) plus the axes as a quaternion (`physics:principalAxes`). Read the
standard way (tensor = R diag R^T), that quaternion gives a tensor with the right eigenvalues but
rotated axes: it is the inverse of the rotation that reproduces the URDF. Mass and centre of mass
are unaffected. Newton/MuJoCo-Warp reads the USD faithfully, so the wrong tensor reaches the solver.

`fix_principal_axes` conjugates each authored quaternion in place, once: the converted layer is
marked (`customLayerData`) so a USD reused from the converter's cache is not flipped back. Needs
`pxr`, imported lazily so the module itself stays importable without Isaac Sim.
"""

from __future__ import annotations

MARKER = "principal_axes_fixed"


def fix_principal_axes(usd_path: str) -> int:
    """Conjugate every authored `physics:principalAxes` in the stage at `usd_path`.

    Returns the number of attributes changed, 0 if the file was already fixed.
    """
    from pxr import Gf, Usd

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetRootLayer()
    data = dict(root.customLayerData)
    if data.get(MARKER):
        return 0

    layers, count = set(), 0
    for prim in stage.Traverse():
        attr = prim.GetAttribute("physics:principalAxes")
        if not attr or not attr.HasAuthoredValue():
            continue
        spec = attr.GetPropertyStack()[0]
        q = spec.default
        if q is None:
            raise RuntimeError(f"{prim.GetPath()}: physics:principalAxes has no default to fix")
        spec.default = Gf.Quatf(q.GetReal(), -q.GetImaginary())
        layers.add(spec.layer)
        count += 1

    if count == 0:
        raise RuntimeError(f"{usd_path}: no physics:principalAxes found; the importer changed")
    data[MARKER] = True
    root.customLayerData = data
    layers.add(root)
    for layer in layers:
        layer.Save()
    return count
