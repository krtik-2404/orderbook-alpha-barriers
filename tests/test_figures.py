"""The one piece of make_figures.py that can be silently wrong.

Everything else in that script draws something a human then looks at. This
function decides which runs are allowed into the three-model comparison, and if
it gets it wrong the figure compares dataset sizes wearing architecture labels -
which looks entirely normal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from make_figures import dataset_windows_k          # noqa: E402


def run(n_per_fold, folds):
    """A run artefact, reduced to the two fields the grouping key reads."""
    return {"folds": [{"n": n} for n in n_per_fold[:folds]]}


def test_same_build_groups_across_different_fold_counts():
    """The real case this exists for: 5 folds of ~109k test windows and 3 folds
    of ~164k are the SAME 657k-window build, split differently. Comparing their
    raw test-row totals (547k vs 493k) would say they are different data."""
    five = run([108882, 109480, 109678, 109479, 109679], 5)
    three = run([163921, 164318, 164518], 3)
    assert dataset_windows_k(five) == dataset_windows_k(three)


def test_a_bigger_build_does_not_group_with_a_smaller_one():
    """September's rebuild is twice the size. If it grouped with August's, the
    figure would put a baseline fitted on 1.16M windows next to a DeepLOB
    fitted on 25k and call the difference architecture."""
    august = run([108882, 109480, 109678, 109479, 109679], 5)
    september = run([232319, 231920, 231324, 231920, 230926], 5)
    assert dataset_windows_k(august) != dataset_windows_k(september)
