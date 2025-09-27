"""Comprehensive tests for the trigger data module."""

import numpy as np

from spine.data.trigger import Trigger


class TestTriggerCreation:
    """Test Trigger object creation and validation."""

    def test_trigger_default(self):
        """Test Trigger creation with default values."""
        trigger = Trigger()
        assert trigger.time_s == -1
        assert trigger.time_ns == -1
        assert trigger.beam_time_s == -1
        assert trigger.beam_time_ns == -1
        assert trigger.type == -1

    def test_trigger_with_values(self):
        """Test Trigger creation with explicit values."""
        trigger = Trigger(
            time_s=1234567890,
            time_ns=123456789,
            beam_time_s=1234567891,
            beam_time_ns=987654321,
            type=1,  # BNB trigger type
        )

        assert trigger.time_s == 1234567890
        assert trigger.time_ns == 123456789
        assert trigger.beam_time_s == 1234567891
        assert trigger.beam_time_ns == 987654321
        assert trigger.type == 1

    def test_trigger_types(self):
        """Test different trigger types."""
        # Common trigger type codes (DAQ-specific)
        trigger_types = [0, 1, 2, 3, 4, 5]  # Different trigger type codes

        for ttype in trigger_types:
            trigger = Trigger(type=ttype)
            assert trigger.type == ttype


class TestTriggerPhysics:
    """Test Trigger physics properties and timing."""

    def test_timing_consistency(self):
        """Test trigger timing consistency."""
        # Test typical timing values
        trigger = Trigger(
            time_s=1234567890,  # Unix timestamp
            time_ns=123456789,  # Nanosecond component
            beam_time_s=1234567890,  # Beam timestamp
            beam_time_ns=500000000,  # Beam nanosecond component
        )

        # Check that timing values are reasonable
        assert trigger.time_s > 1000000000  # After year 2001
        assert 0 <= trigger.time_ns < 1000000000  # Valid nanosecond range
        assert trigger.beam_time_s > 1000000000  # After year 2001
        assert 0 <= trigger.beam_time_ns < 1000000000  # Valid nanosecond range

    def test_beam_timing_relationships(self):
        """Test beam timing relationships."""
        # Beam time should be close to trigger time for beam triggers
        base_time = 1234567890
        trigger_beam = Trigger(
            time_s=base_time,
            time_ns=100000000,
            beam_time_s=base_time,
            beam_time_ns=102000000,  # 2ms later
            type=1,  # BNB trigger type
        )

        # Calculate time difference in nanoseconds
        trigger_total_ns = trigger_beam.time_s * 1e9 + trigger_beam.time_ns
        beam_total_ns = trigger_beam.beam_time_s * 1e9 + trigger_beam.beam_time_ns
        dt = beam_total_ns - trigger_total_ns

        # Beam time should be within reasonable window of trigger time
        assert abs(dt) < 10e9  # Within 10 seconds

    def test_bnb_trigger_scenario(self):
        """Test realistic BNB (Booster Neutrino Beam) trigger scenario."""
        # BNB runs in 1.6 second cycles
        trigger = Trigger(
            time_s=1234567890,
            time_ns=500000000,  # 0.5 seconds into the cycle
            beam_time_s=1234567890,
            beam_time_ns=501800000,  # Beam window starts ~1.8ms later
            type=1,  # BNB trigger type
        )

        assert trigger.type == 1
        # Calculate beam window timing
        trigger_ns = trigger.time_s * 1e9 + trigger.time_ns
        beam_ns = trigger.beam_time_s * 1e9 + trigger.beam_time_ns
        beam_delay = beam_ns - trigger_ns

        # BNB beam typically arrives 1-3 ms after trigger
        assert 1e6 < beam_delay < 5e6  # 1-5 ms in nanoseconds

    def test_numi_trigger_scenario(self):
        """Test realistic NuMI trigger scenario."""
        # NuMI has different timing structure
        trigger = Trigger(
            time_s=1234567890,
            time_ns=200000000,
            beam_time_s=1234567890,
            beam_time_ns=210000000,  # 10ms later
            type=2,  # NuMI trigger type
        )

        assert trigger.type == 2
        # NuMI beam window is different from BNB
        trigger_ns = trigger.time_s * 1e9 + trigger.time_ns
        beam_ns = trigger.beam_time_s * 1e9 + trigger.beam_time_ns
        beam_delay = beam_ns - trigger_ns

        # Check reasonable NuMI timing
        assert beam_delay >= 0  # Beam time after trigger
        assert beam_delay < 50e6  # Within 50ms

    def test_external_trigger_scenario(self):
        """Test external trigger scenario."""
        # External triggers may not have beam timing
        trigger = Trigger(
            time_s=1234567890,
            time_ns=750000000,
            beam_time_s=-1,  # No beam time for external trigger
            beam_time_ns=-1,
            type=3,  # External trigger type
        )

        assert trigger.type == 3
        assert trigger.beam_time_s == -1
        assert trigger.beam_time_ns == -1

    def test_cosmic_trigger_scenario(self):
        """Test cosmic ray trigger scenario."""
        trigger = Trigger(
            time_s=1234567890,
            time_ns=333333333,
            beam_time_s=-1,  # No beam for cosmic triggers
            beam_time_ns=-1,
            type=4,  # Cosmic trigger type
        )

        assert trigger.type == 4
        assert trigger.beam_time_s == -1
        assert trigger.beam_time_ns == -1


class TestTriggerTiming:
    """Test Trigger timing precision and calculations."""

    def test_timing_precision(self):
        """Test trigger timing precision."""
        # Test nanosecond precision - use smaller numbers to avoid float precision loss
        trigger1 = Trigger(time_s=1000, time_ns=100000000)
        trigger2 = Trigger(time_s=1000, time_ns=100000001)
        trigger3 = Trigger(time_s=1001, time_ns=0)

        # Calculate time differences
        t1_total = trigger1.time_s * 1e9 + trigger1.time_ns
        t2_total = trigger2.time_s * 1e9 + trigger2.time_ns
        t3_total = trigger3.time_s * 1e9 + trigger3.time_ns

        assert t2_total - t1_total == 1  # 1 ns difference
        assert t3_total - t1_total == 900000000  # 0.9 s difference

    def test_beam_window_calculations(self):
        """Test beam window timing calculations."""
        triggers = []
        base_time_s = 1234567890

        # Create triggers with different beam delays - use simpler timing
        for i, delay_ms in enumerate([1.0, 1.5, 2.0, 2.5, 3.0]):
            delay_ns = int(delay_ms * 1e6)  # Convert to nanoseconds
            trigger_ns = i * 10000000  # 10ms spacing
            beam_ns = trigger_ns + delay_ns

            trigger = Trigger(
                time_s=base_time_s,
                time_ns=trigger_ns,
                beam_time_s=base_time_s,
                beam_time_ns=beam_ns,
                type=1,  # BNB trigger type
            )
            triggers.append(trigger)

        # Check beam delays are correct
        expected_delays = [1.0, 1.5, 2.0, 2.5, 3.0]  # delays in ms
        for i, trigger in enumerate(triggers):
            expected_delay = expected_delays[i] * 1e6  # Convert to nanoseconds
            actual_delay = trigger.beam_time_ns - trigger.time_ns
            assert abs(actual_delay - expected_delay) < 1000  # Within 1 μs

    def test_trigger_rate_calculations(self):
        """Test trigger rate calculations."""
        # Simulate BNB trigger sequence (1.6s cycle)
        triggers = []
        base_time = 1234567890

        for cycle in range(10):
            trigger = Trigger(
                time_s=base_time + cycle * 2,  # Every 2 seconds (rough approximation)
                time_ns=cycle * 200000000 % 1000000000,
                type=1,  # BNB trigger type
            )
            triggers.append(trigger)

        # Calculate time differences between consecutive triggers
        time_diffs = []
        for i in range(1, len(triggers)):
            t1_ns = triggers[i - 1].time_s * 1e9 + triggers[i - 1].time_ns
            t2_ns = triggers[i].time_s * 1e9 + triggers[i].time_ns
            time_diffs.append(t2_ns - t1_ns)

        # Check that triggers are spaced reasonably
        avg_diff = np.mean(time_diffs)
        assert 1e9 < avg_diff < 5e9  # Between 1-5 seconds average


class TestTriggerTypes:
    """Test different trigger types and their properties."""

    def test_beam_trigger_types(self):
        """Test beam trigger types (BNB=1, NuMI=2)."""
        bnb_trigger = Trigger(
            time_s=1234567890,
            time_ns=100000000,
            beam_time_s=1234567890,
            beam_time_ns=102000000,
            type=1,  # BNB
        )

        numi_trigger = Trigger(
            time_s=1234567890,
            time_ns=200000000,
            beam_time_s=1234567890,
            beam_time_ns=210000000,
            type=2,  # NuMI
        )

        # Both should have valid beam times
        assert bnb_trigger.beam_time_s > 0
        assert bnb_trigger.beam_time_ns >= 0
        assert numi_trigger.beam_time_s > 0
        assert numi_trigger.beam_time_ns >= 0
        assert bnb_trigger.type == 1
        assert numi_trigger.type == 2

    def test_non_beam_trigger_types(self):
        """Test non-beam trigger types."""
        trigger_types = [3, 4, 5, 6]  # EXT, COSMIC, CALIB, TEST

        for ttype in trigger_types:
            trigger = Trigger(time_s=1234567890, time_ns=100000000, type=ttype)
            assert trigger.type == ttype
            # Non-beam triggers may not have beam times
            assert trigger.beam_time_s == -1 or trigger.beam_time_s > 0

    def test_calibration_trigger(self):
        """Test calibration trigger scenario."""
        calib_trigger = Trigger(
            time_s=1234567890,
            time_ns=50000000,
            beam_time_s=-1,  # No beam for calibration
            beam_time_ns=-1,
            type=5,  # Calibration trigger
        )

        assert calib_trigger.type == 5
        assert calib_trigger.beam_time_s == -1
        assert calib_trigger.beam_time_ns == -1

    def test_test_trigger(self):
        """Test trigger for testing purposes."""
        test_trigger = Trigger(
            time_s=1234567890, time_ns=999999999, type=6  # Test trigger
        )

        assert test_trigger.type == 6
        assert test_trigger.time_ns == 999999999  # Max nanosecond value


class TestTriggerIntegration:
    """Test Trigger integration with other components."""

    def test_trigger_serialization(self):
        """Test Trigger object serialization properties."""
        trigger = Trigger(
            time_s=1234567890,
            time_ns=123456789,
            beam_time_s=1234567891,
            beam_time_ns=987654321,
            type=1,  # BNB
        )

        # Test that all attributes are properly set
        assert isinstance(trigger.time_s, int)
        assert isinstance(trigger.time_ns, int)
        assert isinstance(trigger.beam_time_s, int)
        assert isinstance(trigger.beam_time_ns, int)
        assert isinstance(trigger.type, int)

    def test_trigger_collections(self):
        """Test collections of Trigger objects."""
        triggers = []
        trigger_types = [1, 2, 3, 4, 5]  # BNB, NuMI, EXT, COSMIC, CALIB

        for i, ttype in enumerate(trigger_types):
            trigger = Trigger(
                time_s=1234567890 + i,
                time_ns=i * 200000000,
                beam_time_s=1234567890 + i if ttype in [1, 2] else -1,  # BNB, NuMI
                beam_time_ns=(i * 200000000 + 1000000) if ttype in [1, 2] else -1,
                type=ttype,
            )
            triggers.append(trigger)

        assert len(triggers) == 5
        assert all(isinstance(t, Trigger) for t in triggers)

        # Check time ordering
        times = [t.time_s * 1e9 + t.time_ns for t in triggers]
        assert times == sorted(times)

    def test_trigger_edge_cases(self):
        """Test Trigger edge cases and boundary conditions."""
        # Minimum valid trigger
        min_trigger = Trigger(time_s=0, time_ns=0, type=0)
        assert min_trigger.time_s == 0
        assert min_trigger.time_ns == 0

        # Maximum nanosecond value
        max_ns_trigger = Trigger(
            time_s=2147483647,  # Max 32-bit signed int
            time_ns=999999999,  # Max nanosecond value
            type=1,
        )
        assert max_ns_trigger.time_ns == 999999999

        # Default/invalid trigger
        default_trigger = Trigger()
        assert default_trigger.type == -1

    def test_trigger_data_quality(self):
        """Test Trigger data quality indicators."""
        # High quality trigger
        hq_trigger = Trigger(
            time_s=1234567890,
            time_ns=123456789,
            beam_time_s=1234567890,
            beam_time_ns=125456789,
            type=1,  # BNB
        )

        # Low quality trigger (default values)
        lq_trigger = Trigger()

        # Quality indicators
        assert hq_trigger.time_s > 0
        assert hq_trigger.time_ns >= 0
        assert hq_trigger.type > 0
        assert hq_trigger.beam_time_s > 0
        assert hq_trigger.beam_time_ns >= 0

        # Check low quality flags
        assert lq_trigger.time_s < 0
        assert lq_trigger.time_ns < 0
        assert lq_trigger.type < 0
        assert lq_trigger.beam_time_s < 0
        assert lq_trigger.beam_time_ns < 0

    def test_trigger_synchronization(self):
        """Test trigger synchronization scenarios."""
        # Multiple subsystems should see the same trigger
        common_time_s = 1234567890
        common_time_ns = 100000000

        # TPC readout trigger
        tpc_trigger = Trigger(
            time_s=common_time_s, time_ns=common_time_ns, type=1  # BNB
        )

        # PMT readout trigger (same timing)
        pmt_trigger = Trigger(
            time_s=common_time_s, time_ns=common_time_ns, type=1  # BNB
        )

        # CRT readout trigger (same timing)
        crt_trigger = Trigger(
            time_s=common_time_s, time_ns=common_time_ns, type=1  # BNB
        )

        # All triggers should have identical timing
        assert tpc_trigger.time_s == pmt_trigger.time_s == crt_trigger.time_s
        assert tpc_trigger.time_ns == pmt_trigger.time_ns == crt_trigger.time_ns
        assert tpc_trigger.type == pmt_trigger.type == crt_trigger.type

    def test_beam_spill_structure(self):
        """Test beam spill structure timing."""
        # BNB spill structure (multiple bunches)
        base_time_s = 1234567890
        beam_start_ns = 100000000

        # Simulate 81 bunches in a BNB spill (2 μs total length)
        spill_triggers = []
        for bunch in range(81):
            # Each bunch separated by ~25 ns
            bunch_time_ns = beam_start_ns + bunch * 25
            if bunch_time_ns >= 1000000000:
                # Handle nanosecond overflow
                bunch_time_s = base_time_s + 1
                bunch_time_ns = bunch_time_ns - 1000000000
            else:
                bunch_time_s = base_time_s

            trigger = Trigger(
                time_s=base_time_s,
                time_ns=50000000,  # Trigger before beam
                beam_time_s=bunch_time_s,
                beam_time_ns=bunch_time_ns,
                type=1,  # BNB
            )
            spill_triggers.append(trigger)

        # Check that beam times span the expected range
        beam_times = [t.beam_time_s * 1e9 + t.beam_time_ns for t in spill_triggers]
        spill_duration = max(beam_times) - min(beam_times)
        assert 1000 < spill_duration < 3000  # ~2 μs spill duration in nanoseconds
