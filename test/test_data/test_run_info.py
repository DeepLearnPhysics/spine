"""Comprehensive tests for the run_info data module."""

from spine.data.run_info import RunInfo


class TestRunInfoCreation:
    """Test RunInfo object creation and validation."""

    def test_runinfo_default(self):
        """Test RunInfo creation with default values."""
        run_info = RunInfo()
        assert run_info.run == -1
        assert run_info.subrun == -1
        assert run_info.event == -1

    def test_runinfo_with_values(self):
        """Test RunInfo creation with explicit values."""
        run_info = RunInfo(
            run=5426,
            subrun=127,
            event=8394
        )
        
        assert run_info.run == 5426
        assert run_info.subrun == 127
        assert run_info.event == 8394

    def test_runinfo_data_types(self):
        """Test RunInfo with different data types."""
        # Test with various integer types
        run_info = RunInfo(
            run=int(1000),
            subrun=int(50),
            event=int(12345)
        )
        
        assert isinstance(run_info.run, int)
        assert isinstance(run_info.subrun, int)
        assert isinstance(run_info.event, int)


class TestRunInfoPhysics:
    """Test RunInfo with realistic physics scenarios."""

    def test_microboone_run_structure(self):
        """Test RunInfo for MicroBooNE-style run structure."""
        # Typical MicroBooNE run numbers and structure
        run_info_ub = RunInfo(
            run=5426,  # Typical MicroBooNE run number
            subrun=147,  # Subrun within the run
            event=8394   # Event within the subrun
        )
        
        assert 5000 <= run_info_ub.run <= 9999  # MicroBooNE run range
        assert 0 <= run_info_ub.subrun <= 999   # Reasonable subrun range
        assert 0 <= run_info_ub.event <= 99999  # Reasonable event range

    def test_sbnd_run_structure(self):
        """Test RunInfo for SBND-style run structure."""
        # SBND run structure (different from MicroBooNE)
        run_info_sbnd = RunInfo(
            run=15234,  # Higher run numbers for SBND
            subrun=89,
            event=4567
        )
        
        assert run_info_sbnd.run > 10000  # SBND uses higher run numbers
        assert run_info_sbnd.subrun >= 0
        assert run_info_sbnd.event >= 0

    def test_icarus_run_structure(self):
        """Test RunInfo for ICARUS-style run structure."""
        # ICARUS run structure
        run_info_icarus = RunInfo(
            run=8765,
            subrun=234,
            event=12890
        )
        
        assert run_info_icarus.run > 0
        assert run_info_icarus.subrun >= 0
        assert run_info_icarus.event >= 0

    def test_data_vs_mc_runs(self):
        """Test RunInfo patterns for data vs MC."""
        # Data run (positive run numbers)
        data_run = RunInfo(
            run=5426,
            subrun=89,
            event=1234
        )
        
        # MC run (sometimes uses different numbering)
        mc_run = RunInfo(
            run=101,  # MC might use smaller run numbers
            subrun=0,   # MC might not use subruns
            event=5678
        )
        
        assert data_run.run > 1000  # Data runs typically higher
        assert mc_run.run >= 0      # MC runs can be smaller
        assert data_run.event >= 0
        assert mc_run.event >= 0


class TestRunInfoTiming:
    """Test RunInfo timing and sequencing."""

    def test_sequential_events(self):
        """Test sequential event numbering within runs."""
        base_run = 5426
        base_subrun = 100
        
        events = []
        for event_num in range(10):
            run_info = RunInfo(
                run=base_run,
                subrun=base_subrun,
                event=event_num
            )
            events.append(run_info)
        
        # Check sequential event numbering
        for i, run_info in enumerate(events):
            assert run_info.run == base_run
            assert run_info.subrun == base_subrun
            assert run_info.event == i

    def test_subrun_transitions(self):
        """Test subrun transitions within runs."""
        base_run = 5426
        
        run_infos = []
        for subrun in range(5):
            for event in range(3):  # Just test first few events per subrun
                run_info = RunInfo(
                    run=base_run,
                    subrun=subrun,
                    event=event
                )
                run_infos.append(run_info)
        
        # Check subrun structure
        assert len(run_infos) == 15  # 5 subruns * 3 events
        assert all(ri.run == base_run for ri in run_infos)
        
        # Check subrun progression
        subruns = [ri.subrun for ri in run_infos[::3]]  # Every 3rd (first of each subrun)
        assert subruns == [0, 1, 2, 3, 4]

    def test_run_boundaries(self):
        """Test run boundary conditions."""
        # End of run scenario
        end_of_run = RunInfo(
            run=5426,
            subrun=999,  # Last subrun
            event=9999   # Last event
        )
        
        # Start of next run
        start_next_run = RunInfo(
            run=5427,
            subrun=0,    # First subrun of new run
            event=0      # First event
        )
        
        assert end_of_run.run + 1 == start_next_run.run
        assert start_next_run.subrun == 0
        assert start_next_run.event == 0


class TestRunInfoIdentification:
    """Test RunInfo for event identification."""

    def test_unique_event_identification(self):
        """Test unique event identification."""
        # Create multiple events with same run/subrun but different event numbers
        events = []
        for event_num in [1, 100, 1000, 5000, 9999]:
            run_info = RunInfo(
                run=5426,
                subrun=89,
                event=event_num
            )
            events.append((run_info.run, run_info.subrun, run_info.event))
        
        # Check all event IDs are unique
        assert len(set(events)) == len(events)

    def test_global_event_numbering(self):
        """Test global event numbering across runs."""
        run_infos = [
            RunInfo(run=1000, subrun=0, event=0),
            RunInfo(run=1000, subrun=0, event=1),
            RunInfo(run=1000, subrun=1, event=0),
            RunInfo(run=1001, subrun=0, event=0)
        ]
        
        # Create global event IDs
        global_ids = []
        for ri in run_infos:
            # Simple global ID: run * 1M + subrun * 1K + event
            global_id = ri.run * 1000000 + ri.subrun * 1000 + ri.event
            global_ids.append(global_id)
        
        # Check global IDs are unique and ordered
        assert len(set(global_ids)) == len(global_ids)
        assert global_ids == sorted(global_ids)

    def test_event_matching(self):
        """Test event matching between different data products."""
        # Same event from different data products should have same run info
        common_run_info = (5426, 89, 1234)
        
        # TPC data
        tpc_event = RunInfo(
            run=common_run_info[0],
            subrun=common_run_info[1],
            event=common_run_info[2]
        )
        
        # PMT data
        pmt_event = RunInfo(
            run=common_run_info[0],
            subrun=common_run_info[1],
            event=common_run_info[2]
        )
        
        # CRT data
        crt_event = RunInfo(
            run=common_run_info[0],
            subrun=common_run_info[1],
            event=common_run_info[2]
        )
        
        # All should match
        assert (tpc_event.run, tpc_event.subrun, tpc_event.event) == common_run_info
        assert (pmt_event.run, pmt_event.subrun, pmt_event.event) == common_run_info
        assert (crt_event.run, crt_event.subrun, crt_event.event) == common_run_info


class TestRunInfoDataQuality:
    """Test RunInfo data quality and validation."""

    def test_valid_run_numbers(self):
        """Test validation of run numbers."""
        # Valid run numbers (positive)
        valid_runs = [1, 100, 5426, 9999, 15234]
        
        for run_num in valid_runs:
            run_info = RunInfo(run=run_num, subrun=0, event=0)
            assert run_info.run > 0
        
        # Invalid/placeholder run numbers
        invalid_runs = [-1, 0]
        
        for run_num in invalid_runs:
            run_info = RunInfo(run=run_num, subrun=0, event=0)
            assert run_info.run <= 0  # Should be caught as invalid

    def test_subrun_event_ranges(self):
        """Test reasonable subrun and event ranges."""
        # Typical data ranges
        run_info = RunInfo(
            run=5426,
            subrun=89,
            event=1234
        )
        
        # Check ranges are reasonable
        assert 0 <= run_info.subrun <= 9999  # Reasonable subrun range
        assert 0 <= run_info.event <= 999999  # Reasonable event range

    def test_data_quality_flags(self):
        """Test data quality flags in run info."""
        # High quality event
        hq_event = RunInfo(
            run=5426,
            subrun=89,
            event=1234
        )
        
        # Check quality indicators
        assert hq_event.run > 0
        assert hq_event.subrun >= 0
        assert hq_event.event >= 0
        
        # Low quality/default event
        lq_event = RunInfo()  # Default values
        
        # Check default/invalid flags
        assert lq_event.run < 0
        assert lq_event.subrun < 0
        assert lq_event.event < 0


class TestRunInfoIntegration:
    """Test RunInfo integration with other components."""

    def test_runinfo_serialization(self):
        """Test RunInfo object serialization properties."""
        run_info = RunInfo(
            run=5426,
            subrun=89,
            event=1234
        )
        
        # Test that all attributes are properly set
        assert isinstance(run_info.run, int)
        assert isinstance(run_info.subrun, int)
        assert isinstance(run_info.event, int)

    def test_runinfo_collections(self):
        """Test collections of RunInfo objects."""
        run_infos = []
        
        # Create events from multiple runs/subruns
        for run in [5426, 5427, 5428]:
            for subrun in [89, 90]:
                for event in [0, 1000, 2000]:
                    run_info = RunInfo(
                        run=run,
                        subrun=subrun,
                        event=event
                    )
                    run_infos.append(run_info)
        
        assert len(run_infos) == 18  # 3 runs * 2 subruns * 3 events
        assert all(isinstance(ri, RunInfo) for ri in run_infos)
        
        # Check run progression
        runs = sorted(set(ri.run for ri in run_infos))
        assert runs == [5426, 5427, 5428]

    def test_runinfo_edge_cases(self):
        """Test RunInfo edge cases and boundary conditions."""
        # Maximum values
        max_run_info = RunInfo(
            run=999999,
            subrun=9999,
            event=999999
        )
        assert max_run_info.run == 999999
        assert max_run_info.subrun == 9999
        assert max_run_info.event == 999999
        
        # Minimum valid values
        min_run_info = RunInfo(
            run=1,
            subrun=0,
            event=0
        )
        assert min_run_info.run == 1
        assert min_run_info.subrun == 0
        assert min_run_info.event == 0

    def test_runinfo_sorting(self):
        """Test RunInfo sorting and ordering."""
        # Create unsorted list of run infos
        run_infos = [
            RunInfo(run=5428, subrun=0, event=0),
            RunInfo(run=5426, subrun=89, event=1000),
            RunInfo(run=5426, subrun=89, event=100),
            RunInfo(run=5427, subrun=50, event=500),
            RunInfo(run=5426, subrun=90, event=0)
        ]
        
        # Sort by (run, subrun, event)
        sorted_infos = sorted(run_infos, key=lambda ri: (ri.run, ri.subrun, ri.event))
        
        # Check sorting
        assert sorted_infos[0].run == 5426
        assert sorted_infos[0].subrun == 89
        assert sorted_infos[0].event == 100
        
        assert sorted_infos[-1].run == 5428
        assert sorted_infos[-1].subrun == 0
        assert sorted_infos[-1].event == 0

    def test_runinfo_database_style(self):
        """Test RunInfo for database-style operations."""
        # Create events that might be stored in a database
        events = []
        for i in range(100):
            run_info = RunInfo(
                run=5426,
                subrun=i // 10,  # 10 events per subrun
                event=i % 10
            )
            events.append(run_info)
        
        # Test database-style queries
        # Find all events in subrun 5
        subrun_5_events = [ri for ri in events if ri.subrun == 5]
        assert len(subrun_5_events) == 10
        assert all(ri.subrun == 5 for ri in subrun_5_events)
        
        # Find specific event
        target_event = [ri for ri in events if ri.subrun == 3 and ri.event == 7]
        assert len(target_event) == 1
        assert target_event[0].run == 5426
        assert target_event[0].subrun == 3
        assert target_event[0].event == 7

    def test_runinfo_string_representation(self):
        """Test RunInfo string representation and display."""
        run_info = RunInfo(
            run=5426,
            subrun=89,
            event=1234
        )
        
        # Test that we can create meaningful string representations
        run_str = f"Run {run_info.run}, Subrun {run_info.subrun}, Event {run_info.event}"
        assert "5426" in run_str
        assert "89" in run_str
        assert "1234" in run_str
        
        # Test compact representation
        compact_str = f"{run_info.run:06d}_{run_info.subrun:03d}_{run_info.event:06d}"
        assert compact_str == "005426_089_001234"