"""SWE-Together runner for the redo-robustness study (docs/plans/PLAN.md §5, §13).

The benchmark clone lives at ``third_party/swe-together`` (git-ignored, pinned in
``rungs.json``). Every local change to it is kept as one patch file,
``third_party/patches/swe-together/redo_v1.patch``; ``patchset`` writes and
verifies it and every run records its sha256.
"""
