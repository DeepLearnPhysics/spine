"""Comprehensive tests for the optical data module."""

import pytest
import numpy as np

from spine.data.optical import Flash


class TestFlashCreation:
    """Test Flash object creation and validation."""

    def test_flash_default(self):
        """Test Flash creation with default values."""
        flash = Flash()
        assert flash.volume_id == -1
        assert flash.time == -1.0
        assert flash.time_width == -1.0
        assert flash.total_pe == -1.0
        assert len(flash.pe_per_ch) == 0  # Empty array, not None
        # Test default values (these are arrays filled with -inf)
        assert np.allclose(flash.center, [-np.inf, -np.inf, -np.inf])
        assert np.allclose(flash.width, [-np.inf, -np.inf, -np.inf])

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
        np.testing.assert_array_equal(flash.pe_per_ch, pe_per_ch)
        np.testing.assert_array_equal(flash.center, center)
        np.testing.assert_array_equal(flash.width, width)

    def test_flash_numpy_arrays(self):
        """Test Flash with various numpy array types."""
        pe_per_ch = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        width = np.array([5.0, 5.0, 5.0], dtype=np.float64)

        flash = Flash(pe_per_ch=pe_per_ch, center=center, width=width)
        assert flash.pe_per_ch.dtype == np.float32
        assert flash.center.dtype == np.float64
        assert flash.width.dtype == np.float64


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
