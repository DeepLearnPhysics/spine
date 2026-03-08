"""Simple tests for the CRT data module to verify basic functionality and data integrity."""

import numpy as np
import pytest

from spine.data.crt import CRTHit
from spine.utils.conditional import LARCV_AVAILABLE, larcv


class TestCRTHitBasic:
    """Basic CRT hit tests."""

    def test_crthit_default(self):
        """Test CRTHit creation with default values."""
        hit = CRTHit()
        assert hit.id == -1
        assert hit.plane == -1

        assert hit.ts0_s == -1
        assert hit.ts0_ns is np.nan
        assert hit.ts0_s_corr is np.nan
        assert hit.ts0_ns_corr is np.nan
        assert hit.ts1_ns is np.nan
        assert hit.total_pe is np.nan

        assert hit.tagger == ""
        assert hit.units == "cm"

        assert len(hit.feb_id) == 0  # Empty array, not None
        assert np.all(np.isnan(hit.center))  # Array of nan, not None

    def test_crthit_with_values(self):
        """Test CRTHit creation with explicit values."""
        feb_id = np.array([42, 15, 3, 8], dtype=np.ubyte)
        center = np.array([100.0, 50.0, 200.0])

        hit = CRTHit(
            id=123,
            plane=1,
            ts0_s=1234567890,
            ts0_ns=500000000.0,
            ts0_s_corr=1234567890.5,
            ts0_ns_corr=500000100.0,
            ts1_ns=50000.0,
            total_pe=54.7,
            tagger="CRT_Top_01",
            feb_id=feb_id,
            center=center,
        )

        assert hit.id == 123
        assert hit.plane == 1

        assert hit.ts0_s == 1234567890
        assert hit.ts0_ns == 500000000.0
        assert hit.total_pe == 54.7
        assert hit.ts0_s_corr == 1234567890.5
        assert hit.ts0_ns_corr == 500000100.0
        assert hit.ts1_ns == 50000.0

        assert hit.tagger == "CRT_Top_01"
        assert hit.units == "cm"

        np.testing.assert_array_equal(hit.feb_id, feb_id)
        np.testing.assert_array_equal(hit.center, center)

    def test_time_property(self):
        """Test the time property conversion."""
        # Test time property converts ts1_ns to microseconds
        hit = CRTHit(ts1_ns=50000.0)  # 50 microseconds in nanoseconds
        assert hit.time == 50.0  # Should be 50 microseconds

        hit2 = CRTHit(ts1_ns=123456.0)  # 123.456 microseconds in nanoseconds
        assert abs(hit2.time - 123.456) < 1e-10


class TestCRTHitFromLArCV:
    """Tests for CRTHit.from_larcv() - only runs if larcv is available."""

    def test_from_larcv_mock(self):
        """Test from_larcv with mock object (runs even without larcv)."""

        # Create a mock larcv CRTHit object
        class MockLArCVCRTHit:
            """Mock LArCV CRTHit for testing."""

            def id(self):
                return 42

            def plane(self):
                return 3

            def ts0_s(self):
                return 1234567890

            def ts0_ns(self):
                return 500000000.0

            def ts0_s_corr(self):
                return 1234567890.5

            def ts0_ns_corr(self):
                return 500000100.0

            def ts1_ns(self):
                return 75000.0

            def peshit(self):
                return 125.5

            def x_pos(self):
                return 150.0

            def y_pos(self):
                return 250.0

            def z_pos(self):
                return 350.0

            def x_err(self):
                return 5.0

            def y_err(self):
                return 7.5

            def z_err(self):
                return 10.0

            def tagger(self):
                return "CRT_Side_01"

            def feb_id(self):
                return "\x0f\x10\x11\x12"  # 4 bytes

        mock_hit = MockLArCVCRTHit()
        hit = CRTHit.from_larcv(mock_hit)

        # Verify all attributes transferred correctly
        assert hit.id == 42
        assert hit.plane == 3
        assert hit.tagger == "CRT_Side_01"
        assert hit.ts0_s == 1234567890
        assert hit.ts0_ns == 500000000.0
        assert hit.ts0_s_corr == 1234567890.5
        assert hit.ts0_ns_corr == 500000100.0
        assert hit.ts1_ns == 75000.0
        assert hit.total_pe == 125.5

        # Check FEB ID conversion
        np.testing.assert_array_equal(
            hit.feb_id, np.array([15, 16, 17, 18], dtype=np.ubyte)
        )

        # Check position arrays
        np.testing.assert_array_almost_equal(hit.center, [150.0, 250.0, 350.0])
        np.testing.assert_array_almost_equal(hit.width, [5.0, 7.5, 10.0])

    @pytest.mark.skipif(not LARCV_AVAILABLE, reason="larcv not available")
    def test_from_larcv_real(self):
        """Test from_larcv with real larcv object (only if larcv installed)."""
        assert larcv is not None

        # Create a real LArCV CRTHit
        larcv_hit = larcv.CRTHit()
        larcv_hit.id(99)
        larcv_hit.plane(2)
        larcv_hit.tagger("CRT_TestTagger")
        larcv_hit.ts0_s(1234567891)
        larcv_hit.ts0_ns(123456789.0)
        larcv_hit.ts1_ns(50000.0)
        larcv_hit.peshit(200.0)
        larcv_hit.x_pos(100.0)
        larcv_hit.y_pos(200.0)
        larcv_hit.z_pos(300.0)
        larcv_hit.x_err(2.0)
        larcv_hit.y_err(3.0)
        larcv_hit.z_err(4.0)

        # Convert to SPINE CRTHit
        hit = CRTHit.from_larcv(larcv_hit)

        # Verify conversion
        assert hit.id == 99
        assert hit.plane == 2
        assert hit.tagger == "CRT_TestTagger"
        assert hit.ts0_s == 1234567891
        assert hit.ts0_ns == 123456789.0
        assert hit.ts1_ns == 50000.0
        assert hit.total_pe == 200.0
        np.testing.assert_array_almost_equal(hit.center, [100.0, 200.0, 300.0])
        np.testing.assert_array_almost_equal(hit.width, [2.0, 3.0, 4.0])
