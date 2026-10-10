"""Short probes or misrouted URLs must never claim an acceptance pass."""

import pytest

from breakout_rl.laya_benchmark import run_laya_benchmark


@pytest.mark.parametrize('options', [{'samples': 99}, {'rounds': 2}, {'stress': 999}])
def test_requires_full_acceptance_sample_sizes(options):
    with pytest.raises(ValueError, match='Acceptance benchmark requires'):
        run_laya_benchmark(**options)


def test_requires_local_proxy_origin_before_loading_model():
    with pytest.raises(ValueError, match='local HTTP'):
        run_laya_benchmark(url='https://example.com')
