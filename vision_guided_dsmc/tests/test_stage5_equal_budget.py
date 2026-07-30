import numpy as np

from vgdsmc.physical_adaptive import uniform_exact_budget_ppc
from vgdsmc.physical_benchmark import benchmark_cases


def test_uniform_exact_budget_ppc():
    target = uniform_exact_budget_ppc(
        (8, 8),
        base_ppc=10,
        budget_ratio=1.25,
    )
    assert int(target.sum()) == 800
    assert target.min() == 12
    assert target.max() == 13


def test_physical_benchmark_cases_span_knudsen_regimes():
    knudsen = np.array(
        [case.knudsen for case in benchmark_cases()]
    )
    np.testing.assert_allclose(
        knudsen,
        [0.05, 0.10, 0.20],
    )
