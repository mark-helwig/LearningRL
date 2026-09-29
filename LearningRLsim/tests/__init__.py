"""Makes `tests` a package, which is load-bearing for two reasons.

1. The test modules import shared fixtures as `from tests.conftest import ...`.
   Under pytest's default prepend import mode, a package directory causes the
   first NON-package ancestor -- this repo's root -- to be put on `sys.path`,
   so that import resolves no matter where pytest was invoked from. Without
   it, a combined run collecting all three sub-projects at once fails at
   collection.

2. It namespaces these modules as `tests.test_*` rather than bare `test_*`.
   The sibling sub-projects' test directories are not packages, so two of them
   already collide on `test_linting.py` and break that same combined
   collection run. LearningRLsim cannot add a third.
"""
