"""Comprehensive tests for the optical data module."""

import numpy as np
import pytest

from spine.data import Flash
from spine.utils.conditional import LARCV_AVAILABLE, larcv


class TestFlashCreation:
    """Test Flash object creation and validation."""

    def test_flash_default(self):
        """Test Flash creation with default values."""
        flash = Flash()
        assert flash.volume_id == -1
        assert np.isnan(flash.time)
        assert np.isnan(flash.time_width)
        assert np.isnan(flash.total_pe)
        assert len(flash.pe_per_ch) == 0  # Empty array, not None
        # Test default values (these are arrays filled with nan)
        assert np.all(np.isnan(flash.center))
        assert np.all(np.isnan(flash.width))

    def test_flash_with_values(self):
        """Test Flash creation with explicit values."""
        pe_per_ch = np.array([10.5, 20.3, 15.7, 8.2])
        center = np.array([100.0, 50.0, 200.0])
        width = np.array([10.0, 15.0, 12.0])

        flash = Flash(
            volume_id=1,
            time=1500.0,
            time_width=0.1,
            total_pe=54.7,
            pe_per_ch=pe_per_ch,
            center=center,
            width=width,
        )

        assert flash.volume_id == 1
        assert flash.time == 1500.0
        assert flash.time_width == 0.1
        assert flash.total_pe == 54.7
        np.testing.assert_allclose(flash.pe_per_ch, pe_per_ch)
        np.testing.assert_allclose(flash.center, center)
        np.testing.assert_allclose(flash.width, width)

    def test_flash_numpy_arrays(self):
        """Test Flash with various numpy array types.

        Note: Arrays are automatically cast to the dtype specified in the
        field metadata (float32 for Flash arrays), regardless of input dtype.
        """
        pe_per_ch = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        width = np.array([5.0, 5.0, 5.0], dtype=np.float64)

        flash = Flash(pe_per_ch=pe_per_ch, center=center, width=width)
        # All arrays cast to float32 as specified in field metadata
        assert flash.pe_per_ch.dtype == np.float32
        assert flash.center.dtype == np.float32
        assert flash.width.dtype == np.float32


class TestFlashPhysics:
    """Test Flash physics properties and calculations."""

    def test_pe_consistency(self):
        """Test that total PE is consistent with per-detector PE."""
        pe_per_ch = np.array([10.0, 20.0, 30.0, 40.0])
        total_pe = np.sum(pe_per_ch)

        flash = Flash(total_pe=total_pe, pe_per_ch=pe_per_ch)

        # Check consistency
        assert abs(flash.total_pe - np.sum(flash.pe_per_ch)) < 1e-10

    def test_realistic_optical_flash(self):
        """Test realistic optical flash scenario."""
        # Simulating a neutrino interaction flash
        pe_per_ch = np.random.exponential(50.0, 180)  # 180 PMTs typical
        pe_per_ch[pe_per_ch < 0.5] = 0  # Threshold effect

        flash = Flash(
            volume_id=0,  # TPC volume
            time=4500.0,  # Beam time window (μs)
            time_width=0.1,  # Fast scintillation (μs)
            total_pe=np.sum(pe_per_ch),
            pe_per_ch=pe_per_ch,
            center=np.array([0.0, 0.0, 500.0]),  # TPC center
            width=np.array([200.0, 200.0, 400.0]),  # Flash extent
        )

        assert flash.volume_id == 0
        assert 4000 < flash.time < 5000  # Beam window
        assert flash.time_width < 1.0  # Fast component
        assert flash.total_pe > 0
        assert len(flash.pe_per_ch) == 180
        assert flash.center.shape == (3,)
        assert flash.width.shape == (3,)

    def test_cosmic_flash(self):
        """Test cosmic ray flash scenario."""
        # Cosmic flashes are typically longer and less intense
        pe_per_ch = np.random.exponential(10.0, 180)
        pe_per_ch[pe_per_ch < 0.5] = 0

        flash = Flash(
            volume_id=0,
            time=2000.0,  # Random time outside beam window
            time_width=1.5,  # Longer scintillation
            total_pe=np.sum(pe_per_ch),
            pe_per_ch=pe_per_ch,
            center=np.array([100.0, -50.0, 300.0]),
            width=np.array([400.0, 300.0, 800.0]),  # More extended
        )

        assert flash.time_width > 1.0  # Cosmic characteristic
        assert flash.total_pe < 5000  # Typically less PE than neutrino
        assert np.all(flash.width > 200.0)  # More extended

    def test_flash_timing(self):
        """Test flash timing properties."""
        # Test various timing scenarios
        flash_beam = Flash(time=4500.0, time_width=0.1)  # Beam flash
        flash_cosmic = Flash(time=1000.0, time_width=2.0)  # Cosmic flash
        flash_late = Flash(time=8000.0, time_width=0.5)  # Late flash

        assert 4000 < flash_beam.time < 5000
        assert flash_beam.time_width < 0.5
        assert flash_cosmic.time < 3000
        assert flash_cosmic.time_width > 1.0
        assert flash_late.time > 7000


class TestFlashDetectorGeometry:
    """Test Flash interaction with detector geometry."""

    def test_detector_volumes(self):
        """Test flashes in different detector volumes."""
        # Test different TPC volumes
        flash_tpc0 = Flash(volume_id=0, center=np.array([0.0, 0.0, 500.0]))
        flash_tpc1 = Flash(volume_id=1, center=np.array([250.0, 0.0, 500.0]))

        assert flash_tpc0.volume_id == 0
        assert flash_tpc1.volume_id == 1
        assert flash_tpc0.center[0] != flash_tpc1.center[0]

    def test_optical_detector_coverage(self):
        """Test optical detector coverage scenarios."""
        # Full detector coverage
        pe_full = np.random.exponential(20.0, 180)
        np.random.seed(42)  # Set seed for reproducible test
        pe_full = np.random.exponential(20.0, 180)
        flash_full = Flash(pe_per_ch=pe_full)

        # Partial coverage (some PMTs see no light)
        pe_partial = pe_full.copy()
        pe_partial[pe_partial < 5.0] = 0.0
        flash_partial = Flash(pe_per_ch=pe_partial)

        # Localized flash (only few PMTs)
        pe_local = np.zeros(180)
        pe_local[50:60] = np.random.exponential(100.0, 10)
        flash_local = Flash(pe_per_ch=pe_local)

        assert np.sum(flash_full.pe_per_ch > 0) >= 80  # Most PMTs see light
        assert np.sum(flash_partial.pe_per_ch > 0) < np.sum(flash_full.pe_per_ch > 0)
        assert np.sum(flash_local.pe_per_ch > 0) == 10

    def test_flash_reconstruction_quality(self):
        """Test flash reconstruction quality indicators."""
        # High quality flash (well reconstructed)
        high_pe = np.random.exponential(100.0, 180)
        high_pe[high_pe < 1.0] = 0
        flash_hq = Flash(
            total_pe=np.sum(high_pe),
            pe_per_ch=high_pe,
            center=np.array([0.0, 0.0, 500.0]),
            width=np.array([50.0, 50.0, 100.0]),  # Well localized
        )

        # Low quality flash (poorly reconstructed)
        low_pe = np.random.exponential(5.0, 180)
        low_pe[low_pe < 0.5] = 0
        flash_lq = Flash(
            total_pe=np.sum(low_pe),
            pe_per_ch=low_pe,
            center=np.array([0.0, 0.0, 500.0]),
            width=np.array([300.0, 300.0, 600.0]),  # Poorly localized
        )

        assert flash_hq.total_pe > flash_lq.total_pe
        assert np.all(flash_hq.width < flash_lq.width)


class TestFlashCollections:
    """Test Flash collections and matching."""

    def test_multiple_flashes(self):
        """Test handling multiple flashes."""
        flashes = []
        for i in range(5):
            pe_per_ch = np.random.exponential(30.0, 180)
            pe_per_ch[pe_per_ch < 1.0] = 0

            flash = Flash(
                volume_id=i % 2,
                time=4500.0 + i * 0.5,
                time_width=0.1 + i * 0.02,
                total_pe=np.sum(pe_per_ch),
                pe_per_ch=pe_per_ch,
                center=np.array([i * 10.0, 0.0, 500.0]),
                width=np.array([50.0, 50.0, 100.0]),
            )
            flashes.append(flash)

        assert len(flashes) == 5
        assert all(isinstance(f, Flash) for f in flashes)

        # Check time ordering
        times = [f.time for f in flashes]
        assert times == sorted(times)

    def test_flash_matching_criteria(self):
        """Test flash matching with different criteria."""
        # Primary flash (brightest)
        primary_flash = Flash(
            time=4500.0, total_pe=1000.0, center=np.array([0.0, 0.0, 500.0])
        )

        # Secondary flash (dimmer, later)
        secondary_flash = Flash(
            time=4500.5, total_pe=200.0, center=np.array([50.0, 0.0, 500.0])
        )

        # Background flash (early, dim)
        background_flash = Flash(
            time=1000.0, total_pe=50.0, center=np.array([0.0, 0.0, 500.0])
        )

        flashes = [primary_flash, secondary_flash, background_flash]

        # Test PE-based selection
        brightest = max(flashes, key=lambda f: f.total_pe)
        assert brightest is primary_flash

        # Test time-based selection (beam window)
        beam_flashes = [f for f in flashes if 4000 < f.time < 5000]
        assert len(beam_flashes) == 2
        assert background_flash not in beam_flashes


class TestFlashIntegration:
    """Test Flash integration with other components."""

    def test_flash_serialization(self):
        """Test Flash object serialization properties."""
        pe_per_ch = np.array([1.0, 2.0, 3.0, 4.0])
        center = np.array([10.0, 20.0, 30.0])
        width = np.array([5.0, 10.0, 15.0])

        flash = Flash(
            volume_id=1,
            time=1500.0,
            time_width=0.1,
            total_pe=10.0,
            pe_per_ch=pe_per_ch,
            center=center,
            width=width,
        )

        # Test that arrays maintain their properties
        assert isinstance(flash.pe_per_ch, np.ndarray)
        assert isinstance(flash.center, np.ndarray)
        assert isinstance(flash.width, np.ndarray)
        assert flash.pe_per_ch.shape == (4,)
        assert flash.center.shape == (3,)
        assert flash.width.shape == (3,)

    def test_flash_edge_cases(self):
        """Test Flash edge cases and boundary conditions."""
        # Empty flash
        empty_flash = Flash(total_pe=0.0, pe_per_ch=np.zeros(180))
        assert empty_flash.total_pe == 0.0
        assert np.all(empty_flash.pe_per_ch == 0.0)

        # Single PMT flash
        single_pe = np.zeros(180)
        single_pe[90] = 100.0
        single_flash = Flash(total_pe=100.0, pe_per_ch=single_pe)
        assert single_flash.total_pe == 100.0
        assert np.sum(single_flash.pe_per_ch > 0) == 1

        # Very bright flash
        bright_pe = np.full(180, 1000.0)
        bright_flash = Flash(total_pe=180000.0, pe_per_ch=bright_pe)
        assert bright_flash.total_pe == 180000.0
        assert np.all(bright_flash.pe_per_ch == 1000.0)


class TestFlashMerging:
    """Test merging of Flash objects."""

    def test_flash_merging_rejects_mismatched_units(self):
        """Test that flashes with inconsistent units cannot be merged."""
        flash_cm = Flash(units="cm")
        flash_px = Flash(units="px")

        with pytest.raises(ValueError, match="units of the flash"):
            flash_cm.merge(flash_px)

    def test_flash_merging(self):
        """Test merging two Flash objects."""
        flash1 = Flash(
            time=4500.0,
            time_width=0.1,
            total_pe=100.0,
            pe_per_ch=np.array([10.0, 20.0, 30.0, 40.0]),
            center=np.array([0.0, 0.0, 500.0]),
            width=np.array([50.0, 50.0, 100.0]),
            volume_id=0,
        )

        flash2 = Flash(
            time=4500.5,
            time_width=0.2,
            total_pe=200.0,
            pe_per_ch=np.array([20.0, 30.0, 40.0, 50.0]),
            center=np.array([50.0, 0.0, 500.0]),
            width=np.array([60.0, 60.0, 120.0]),
            volume_id=1,
        )

        # Create a merged flash starting with flash1 and merging flash2 into it
        merged_flash = Flash(
            time=flash1.time,  # Will be updated in merge
            time_width=flash1.time_width,  # Will be updated in merge
            total_pe=flash1.total_pe,  # Will be updated in merge
            pe_per_ch=flash1.pe_per_ch.copy(),  # Will be updated in merge
            center=flash1.center.copy(),  # Will be updated in merge
            width=flash1.width.copy(),  # Will be updated in merge
            volume_id=flash1.volume_id,
        )

        merged_flash.merge(flash2)

        # Check on timing information (earlier flash should dominate)
        assert merged_flash.time == flash1.time
        assert (
            merged_flash.time_width
            == max(flash1.time + flash1.time_width, flash2.time + flash2.time_width)
            - merged_flash.time
        )

        # Check on photoelectron consistency
        merged_pe = flash1.pe_per_ch + flash2.pe_per_ch
        merged_center = (
            flash1.center / flash1.width**2 + flash2.center / flash2.width**2
        ) / (1.0 / flash1.width**2 + 1.0 / flash2.width**2)
        merged_width = 1.0 / np.sqrt(1.0 / flash1.width**2 + 1.0 / flash2.width**2)

        assert merged_flash.total_pe == flash1.total_pe + flash2.total_pe
        np.testing.assert_allclose(merged_flash.pe_per_ch, merged_pe)
        np.testing.assert_allclose(merged_flash.center, merged_center)
        np.testing.assert_allclose(merged_flash.width, merged_width)

        # Now merge in the opposite direction and check that timing information updates correctly
        merged_flash2 = Flash(
            time=flash2.time,  # Will be updated in merge
            time_width=flash2.time_width,  # Will be updated in merge
            total_pe=flash2.total_pe,  # Will be updated in merge
            pe_per_ch=flash2.pe_per_ch.copy(),  # Will be updated in merge
            center=flash2.center.copy(),  # Will be updated in merge
            width=flash2.width.copy(),  # Will be updated in merge
            volume_id=flash2.volume_id,
        )
        merged_flash2.merge(flash1)

        assert merged_flash2.time == flash1.time
        assert (
            merged_flash2.time_width
            == max(flash1.time + flash1.time_width, flash2.time + flash2.time_width)
            - merged_flash2.time
        )

        assert merged_flash2.total_pe == flash1.total_pe + flash2.total_pe
        np.testing.assert_allclose(merged_flash2.pe_per_ch, merged_pe)
        np.testing.assert_allclose(merged_flash2.center, merged_center)
        np.testing.assert_allclose(merged_flash2.width, merged_width)


class TestFlashFromLArCV:
    """Tests for Flash.from_larcv() - only runs if larcv is available."""

    def test_from_larcv_mock(self):
        """Test from_larcv with mock object (runs even without larcv)."""

        # Create a mock larcv Flash object
        class MockLArCVFlash:
            """Mock LArCV Flash for testing."""

            def id(self):
                return 5

            def volume_id(self):
                return 0

            def frame(self):
                return 100

            def inBeamFrame(self):
                return True

            def onBeamTime(self):
                return 1

            def time(self):
                return 4.5  # us

            def absTime(self):
                return 1234567.0

            def timeWidth(self):
                return 0.5  # us

            def TotalPE(self):
                return 350.5

            def xCenter(self):
                return 125.0

            def yCenter(self):
                return 25.0

            def zCenter(self):
                return 500.0

            def xWidth(self):
                return 10.0

            def yWidth(self):
                return 15.0

            def zWidth(self):
                return 20.0

            def PEPerOpDet(self):
                # Return a mock list of PE per optical detector (e.g., 10 PMTs)
                return [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 30.0, 20.0, 20.5]

        mock_flash = MockLArCVFlash()
        flash = Flash.from_larcv(mock_flash)

        # Verify all attributes transferred correctly
        assert flash.id == 5
        assert flash.volume_id == 0
        assert flash.frame == 100
        assert flash.in_beam_frame is True
        assert flash.on_beam_time == 1
        assert flash.time == 4.5
        assert flash.time_abs == 1234567.0
        assert flash.time_width == 0.5
        assert flash.total_pe == 350.5

        # Check position and width arrays
        np.testing.assert_array_almost_equal(flash.center, [125.0, 25.0, 500.0])
        np.testing.assert_array_almost_equal(flash.width, [10.0, 15.0, 20.0])

        # Check PE per channel
        expected_pe = np.array(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 30.0, 20.0, 20.5]
        )
        np.testing.assert_array_almost_equal(flash.pe_per_ch, expected_pe)

    @pytest.mark.skipif(not LARCV_AVAILABLE, reason="larcv not available")
    def test_from_larcv_real(self):
        """Test from_larcv with real larcv object (only if larcv installed)."""
        assert larcv is not None

        # Create a real LArCV Flash
        larcv_flash = larcv.Flash()
        larcv_flash.id(8)
        larcv_flash.volume_id(1)
        larcv_flash.frame(200)
        larcv_flash.inBeamFrame(True)
        larcv_flash.onBeamTime(0)
        larcv_flash.time(3.8)
        larcv_flash.absTime(9876543.0)
        larcv_flash.timeWidth(0.75)

        # Set position (xCenter, yCenter, zCenter)
        larcv_flash.xCenter(150.0)
        larcv_flash.yCenter(50.0)
        larcv_flash.zCenter(600.0)

        # Set width (xWidth, yWidth, zWidth)
        larcv_flash.xWidth(12.0)
        larcv_flash.yWidth(18.0)
        larcv_flash.zWidth(25.0)

        # Set PE per optical detector
        pe_list = [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 50.0, 50.0]
        larcv_flash.PEPerOpDet(pe_list)

        # Convert to SPINE Flash
        flash = Flash.from_larcv(larcv_flash)

        # Verify conversion
        assert flash.id == 8
        assert flash.volume_id == 1
        assert flash.frame == 200
        assert flash.in_beam_frame is True
        assert flash.on_beam_time == 0
        assert flash.time == 3.8
        assert flash.time_abs == 9876543.0
        assert flash.time_width == 0.75
        assert flash.total_pe == np.sum(pe_list)

        np.testing.assert_array_almost_equal(flash.center, [150.0, 50.0, 600.0])
        np.testing.assert_array_almost_equal(flash.width, [12.0, 18.0, 25.0])

        expected_pe = np.array(pe_list, dtype=np.float32)
        np.testing.assert_array_almost_equal(flash.pe_per_ch, expected_pe)
