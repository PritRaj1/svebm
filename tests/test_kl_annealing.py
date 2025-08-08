import pytest
import numpy as np

from src.variational.kl_annealing import KLAnnealer

class TestKLAnnealer:

    @pytest.fixture
    def basic_annealer(self):
        return KLAnnealer(
            total_steps=1000,
            n_cycle=2,
            ratio_increase=0.25,
            ratio_zero=0.5,
            max_kl_weight=1.0,
            min_kl_weight=0.0,
        )

    def test_annealer_instantiation(self, basic_annealer):
        assert basic_annealer.total_steps == 1000
        assert basic_annealer.n_cycle == 2
        assert basic_annealer.ratio_increase == 0.25
        assert basic_annealer.ratio_zero == 0.5
        assert basic_annealer.max_kl_weight == 1.0
        assert basic_annealer.min_kl_weight == 0.0
        assert len(basic_annealer._schedule) == 1000

    def test_schedule_phases(self, basic_annealer):
        """Verify zero/increase/max phases per cycle."""
        sched = basic_annealer._schedule
        period = int(basic_annealer.total_steps / basic_annealer.n_cycle)

        zero_end = int(period * basic_annealer.ratio_zero)
        inc_end = int(period * (basic_annealer.ratio_zero + basic_annealer.ratio_increase))

        zero_seg = sched[:zero_end]
        inc_seg = sched[zero_end:inc_end]
        max_seg = sched[inc_end:period]

        assert np.allclose(zero_seg, basic_annealer.min_kl_weight, atol=1e-8)
        if len(inc_seg) > 1:
            assert np.all(np.diff(inc_seg) > 0)
        assert np.allclose(max_seg, basic_annealer.max_kl_weight, atol=1e-8)

        start = period
        zero_seg2 = sched[start:start + zero_end]
        inc_seg2 = sched[start + zero_end:start + inc_end]
        max_seg2 = sched[start + inc_end:start + period]

        assert np.allclose(zero_seg2, basic_annealer.min_kl_weight, atol=1e-8)
        if len(inc_seg2) > 1:
            assert np.all(np.diff(inc_seg2) > 0)
        assert np.allclose(max_seg2, basic_annealer.max_kl_weight, atol=1e-8)
