"""Mesh reference auditing and raw STL stats. Exactly three functions, no more.

No `trimesh`, no decimation, no format conversion, no convex decomposition --
the Isaac Lab importer already does convex hulls and nothing else was asked
for. If any of these three functions turns out unused once G1's pipeline is
green end to end, delete it rather than grow it.

Pure numpy + stdlib (`xml.etree`, `struct`) -- no Isaac Lab dependency, so this
runs on Windows with no vendor/ checkout, same as robot_schema.py.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MeshRef:
    """One `<mesh filename="...">` reference found in a URDF/Xacro-style
    description file."""

    raw_filename: str  # exactly as written in the XML, e.g. "meshes/pelvis.STL"
    resolved_path: Path  # absolute path this reference is expected to resolve to
    link_name: str  # the owning <link> element's name
    kind: str  # "visual" or "collision"


@dataclass(frozen=True)
class MeshAudit:
    """Result of auditing a list of `MeshRef` against the filesystem."""

    missing: tuple[MeshRef, ...]
    case_mismatches: tuple[tuple[MeshRef, Path], ...]  # (ref, path that DOES exist, wrong case)
    duplicates: tuple[str, ...]  # resolved paths referenced more than once
    total_bytes: int  # sum over unique existing files
    largest: tuple[tuple[Path, int], ...]  # (path, bytes), largest-first, top N

    @property
    def ok(self) -> bool:
        return not self.missing and not self.case_mismatches


@dataclass(frozen=True)
class MeshStats:
    """Geometric summary of one binary or ASCII STL file, from a pure-numpy
    reader -- no `trimesh`."""

    n_triangles: int
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
    is_closed: bool  # every edge shared by exactly two triangles (watertight proxy)
    signed_volume: float  # divergence-theorem volume; sign indicates outward-normal convention


def iter_mesh_refs(description_path: str | Path) -> list[MeshRef]:
    """Every `<mesh filename=...>` in a URDF, with the filename resolved to an
    absolute path, the owning `<link>` name, and whether it came from
    `<visual>` or `<collision>`.

    This is the only place in the repo that knows how mesh paths resolve:
    relative to the description file's own directory (true for
    `unitree_ros` -- F2: relative mesh references, `ros_package_paths` stays
    empty) or, if the filename starts with `package://`, relative to the
    first matching entry in `ros_package_paths` (kept general even though the
    Unitree assets never need it).
    """
    description_path = Path(description_path).resolve()
    base_dir = description_path.parent

    tree = ET.parse(description_path)
    root = tree.getroot()

    refs: list[MeshRef] = []
    for link in root.iter("link"):
        link_name = link.get("name", "<unnamed link>")
        for kind, tag in (("visual", "visual"), ("collision", "collision")):
            for geom_parent in link.findall(tag):
                mesh = geom_parent.find("./geometry/mesh")
                if mesh is None:
                    continue
                filename = mesh.get("filename")
                if not filename:
                    continue
                resolved = _resolve_mesh_filename(filename, base_dir)
                refs.append(
                    MeshRef(
                        raw_filename=filename,
                        resolved_path=resolved,
                        link_name=link_name,
                        kind=kind,
                    )
                )
    return refs


def _resolve_mesh_filename(filename: str, base_dir: Path) -> Path:
    if filename.startswith("package://"):
        # Not exercised by any vendored asset here (F2a/F2b: relative refs only,
        # ros_package_paths stays empty) but kept general: strip the package
        # name and resolve relative to base_dir, since we have no ROS package
        # index to consult and this repo never sets ros_package_paths.
        remainder = filename[len("package://") :]
        _, _, rel = remainder.partition("/")
        return (base_dir / rel).resolve()
    return (base_dir / filename).resolve()


def check_mesh_refs(refs: list[MeshRef], top_n: int = 10) -> MeshAudit:
    """Audit mesh references against the filesystem.

    Exact-case comparison is the whole point: Unitree ships uppercase `.STL`
    files, and a URDF referencing `.stl` resolves fine on case-insensitive
    Windows/macOS but fails outright on case-sensitive Linux. Deliberately
    does NOT compare against `MeshRef.resolved_path.name`, since
    `Path.resolve()` silently rewrites casing to match disk on a
    case-insensitive filesystem -- masking the very mismatch this catches.
    Uses `MeshRef.raw_filename` (untouched by `resolve()`) against an
    exact-case directory listing instead.
    """
    missing: list[MeshRef] = []
    case_mismatches: list[tuple[MeshRef, Path]] = []
    seen_resolved: dict[str, int] = {}
    sizes: dict[Path, int] = {}

    dir_listings_exact: dict[Path, dict[str, Path]] = {}

    for ref in refs:
        resolved = ref.resolved_path
        key = str(resolved).lower()
        seen_resolved[key] = seen_resolved.get(key, 0) + 1

        # As the URDF/XML actually wrote it, not as `resolve()` rewrote it.
        written_name = ref.raw_filename.replace("\\", "/").rsplit("/", 1)[-1]

        parent = resolved.parent
        if parent not in dir_listings_exact:
            if parent.is_dir():
                dir_listings_exact[parent] = {p.name: p for p in parent.iterdir()}
            else:
                dir_listings_exact[parent] = {}
        listing = dir_listings_exact[parent]

        exact_match = listing.get(written_name)
        if exact_match is not None:
            sizes[exact_match] = exact_match.stat().st_size
            continue

        case_insensitive_match = next(
            (p for name, p in listing.items() if name.lower() == written_name.lower()),
            None,
        )
        if case_insensitive_match is not None:
            case_mismatches.append((ref, case_insensitive_match))
            sizes[case_insensitive_match] = case_insensitive_match.stat().st_size
        else:
            missing.append(ref)

    duplicates = tuple(key for key, count in seen_resolved.items() if count > 1)
    total_bytes = sum(sizes.values())
    largest = tuple(sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:top_n])

    return MeshAudit(
        missing=tuple(missing),
        case_mismatches=tuple(case_mismatches),
        duplicates=duplicates,
        total_bytes=total_bytes,
        largest=largest,
    )


def stl_stats(path: str | Path) -> MeshStats:
    """Triangle count, AABB extents, a watertight-ness proxy, and signed
    volume for one STL file -- binary or ASCII, auto-detected.

    Pure numpy, ~60 lines, no `trimesh` (the dependency set stays Isaac Lab +
    Newton only, per project rule). AABB extents catch a metre/millimetre
    unit error in a vendor mesh -- otherwise invisible until the robot
    explodes at t=0 in simulation. Signed volume divided by the link's mass
    (from the URDF `<inertial>`) gives an implied density, which is a cheap
    smell test for "this mesh and this mass don't belong together."
    """
    path = Path(path)
    triangles = _read_stl_triangles(path)  # (n_triangles, 3, 3) float64

    n_triangles = triangles.shape[0]
    all_vertices = triangles.reshape(-1, 3)
    aabb_min = tuple(float(v) for v in all_vertices.min(axis=0))
    aabb_max = tuple(float(v) for v in all_vertices.max(axis=0))

    # Signed volume via the divergence theorem: sum over triangles of
    # (1/6) * v0 . (v1 x v2). For a closed, consistently-wound mesh this is
    # the exact enclosed volume (sign = outward-normal convention: positive
    # for outward-facing normals in a right-handed winding).
    v0, v1, v2 = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    cross = np.cross(v1, v2)
    signed_volume = float(np.einsum("ij,ij->i", v0, cross).sum() / 6.0)

    is_closed = _is_watertight(triangles)

    return MeshStats(
        n_triangles=n_triangles,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        is_closed=is_closed,
        signed_volume=signed_volume,
    )


def _read_stl_triangles(path: Path):
    """Return an (N, 3, 3) numpy array of triangle vertices, from either STL
    flavor. Binary STL is detected by the standard heuristic: an 80-byte
    header, a uint32 triangle count, and a file size matching
    `84 + 50 * n_triangles` exactly. Otherwise treated as ASCII STL."""
    raw = path.read_bytes()

    if len(raw) >= 84:
        (declared_n,) = struct.unpack_from("<I", raw, 80)
        if 84 + 50 * declared_n == len(raw):
            triangles = np.empty((declared_n, 3, 3), dtype=np.float64)
            offset = 84
            for i in range(declared_n):
                # 12 bytes normal (skipped), 3x12 bytes vertices, 2 bytes attr
                verts = struct.unpack_from("<9f", raw, offset + 12)
                triangles[i, 0, :] = verts[0:3]
                triangles[i, 1, :] = verts[3:6]
                triangles[i, 2, :] = verts[6:9]
                offset += 50
            return triangles

    # ASCII STL: "vertex x y z" lines, three per "facet".
    text = raw.decode("ascii", errors="replace")
    verts: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("vertex"):
            parts = stripped.split()
            if len(parts) != 4:
                continue
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts or len(verts) % 3 != 0:
        raise ValueError(f"{path}: could not parse as binary or ASCII STL")
    arr = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    return arr


def _is_watertight(triangles) -> bool:
    """Proxy for "closed mesh": every undirected edge must be shared by
    exactly two triangles. Not a full manifold check (does not verify
    consistent winding), but cheap and catches the common vendor-mesh defect
    of a hole or a dangling triangle."""
    edge_counts: dict[tuple[tuple[float, float, float], tuple[float, float, float]], int] = {}
    for tri in triangles:
        pts = [tuple(round(float(c), 6) for c in v) for v in tri]
        for a, b in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])):
            edge = (a, b) if a < b else (b, a)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return all(count == 2 for count in edge_counts.values())
