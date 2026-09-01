"""Metric-specific visualization and report-building helpers.

The package namespace intentionally does not eagerly re-export its submodules.
Import concrete helpers from :mod:`spine.vis.metric.confmat`,
:mod:`spine.vis.metric.distribution`, :mod:`spine.vis.metric.heatmap` or
:mod:`spine.vis.metric.report`. Explicit imports keep function ownership clear
and prevent similarly named visualization helpers from shadowing one another.
"""
