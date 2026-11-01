# external/

Third-party code that mimic projects build on, fetched by a second gitman
config (the repo-root `gitman.yml` only vendors robot assets into
`robot_description/vendor/`).

```bash
gitman install --root external    # checkouts land at external/<name>/, gitignored
gitman lock --root external       # after changing a `rev:`; rewrites gitman.yml, drops comments
```

## Pins

| Name | Repo | Pinned | Why |
|---|---|---|---|
| `GMR` | [Roboparty/GMR](https://github.com/Roboparty/GMR) | `f7bfaea` (`main`) | General Motion Retargeting: human motion (SMPL-X, BVH) to robot joint trajectories via IK, for the mimic projects. RoboParty's fork, so the RPO is a supported robot. |

Install a checkout into your env with `pip install -e external/GMR`. GMR
targets Python 3.10 upstream and pulls `mink`, `qpsolvers`, `smplx` (from git).
It is not a dependency of any sub-project here.
