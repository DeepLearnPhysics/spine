"""Simple tests for the CRT data module to fix the failing cases."""

import numpy as np

from spine.data.crt import CRTHit


class TestCRTHitBasic:
    """Basic CRT hit tests."""

    def test_crthit_default(self):
        """Test CRTHit creation with default values."""
        hit = CRTHit()
        assert hit.plane == -1
        assert hit.tagger == ""
        assert len(hit.feb_id) == 0  # Empty array, not None
        assert hit.ts0_s == -1
        assert hit.ts0_ns == -1.0
        assert hit.total_pe == -1.0
        assert np.all(~np.isfinite(hit.center))  # Array of -inf, not None

    def test_crthit_with_values(self):
        """Test CRTHit creation with explicit values."""
        feb_id = np.array([42, 15, 3, 8], dtype=np.ubyte)
        center = np.array([100.0, 50.0, 200.0])
        
        hit = CRTHit(
            plane=1,
            tagger="CRT_Top_01",
            feb_id=feb_id,
            ts0_s=1234567890,
            ts0_ns=500000000.0,
            total_pe=54.7,
            center=center
        )
        
        assert hit.plane == 1
        assert hit.tagger == "CRT_Top_01"
        assert hit.ts0_s == 1234567890
        assert hit.ts0_ns == 500000000.0
        assert hit.total_pe == 54.7
        np.testing.assert_array_equal(hit.feb_id, feb_id)
        np.testing.assert_array_equal(hit.center, center)

    def test_crthit_data_quality(self):
        """Test CRTHit data quality indicators."""
        # High quality hit (good timing, reasonable PE)
        feb_id_hq = np.array([15, 16, 17, 18], dtype=np.ubyte)
        hq_hit = CRTHit(
            plane=0,
            tagger="CRT_Top_01",
            feb_id=feb_id_hq,
            ts0_s=1234567890,
            ts0_ns=123456789.0,
            ts1_ns=50000.0,
            total_pe=150.0,
            center=np.array([10.0, 300.0, 50.0])
        )
        
        # Low quality hit (default/invalid parameters)
        lq_hit = CRTHit()
        
        # Quality indicators
        assert hq_hit.plane >= 0
        assert hq_hit.tagger != ""
        assert len(hq_hit.feb_id) > 0  # Has FEB data
        assert hq_hit.ts0_s > 0
        assert hq_hit.ts0_ns >= 0
        assert hq_hit.total_pe > 0
        assert hq_hit.center is not None
        
        # Check low quality flags
        assert lq_hit.plane < 0
        assert lq_hit.tagger == ""
        assert len(lq_hit.feb_id) == 0  # Empty array
        assert lq_hit.ts0_s < 0
        assert lq_hit.ts0_ns < 0
        assert lq_hit.total_pe < 0

    def test_time_property(self):
        """Test the time property conversion."""
        # Test time property converts ts1_ns to microseconds
        hit = CRTHit(ts1_ns=50000.0)  # 50 microseconds in nanoseconds
        assert hit.time == 50.0  # Should be 50 microseconds
        
        hit2 = CRTHit(ts1_ns=123456.0)  # 123.456 microseconds in nanoseconds
        assert abs(hit2.time - 123.456) < 1e-10