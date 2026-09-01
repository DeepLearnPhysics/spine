# Full-chain metric reporting

`report_240805.yaml` reduces the CSV files written by the full-chain metric
analyzers after every inference shard has completed. The reporter is a separate
CPU-only process; it does not initialize the SPINE `Driver`, Torch, or LArCV.

Run it with:

```bash
spine-report \
  --config config/test/generic/full_chain/report_240805.yaml \
  --input-dir "$workspace/metrics/full_chain/raw" \
  --output-dir "$workspace/metrics/full_chain/report"
```

The input patterns are evaluated recursively beneath `--input-dir`. In strict
mode, every configured input must match at least one completed CSV shard. The
report contains the discovered source paths and counts, configuration checksum,
configured dataset/checkpoint provenance, metric schema version, reduced metric
values, and plots. If a checkpoint path is configured without a checksum, the
reporter calculates its SHA-256 digest while streaming the file.

The scheduler should submit this command as one small CPU job with an `afterok`
dependency on every inference array or chunk job that contributes CSVs. The
reporter does not contain scheduler-specific logic.

PPN distances are streamed in configurable chunks. `distance_scale` can be used
when the analyzer distance unit needs conversion before thresholds and
histograms are evaluated. Clustering metric ranges are configurable through a
`metric_ranges` mapping; ARI defaults to `[-1, 1]`, while efficiency and purity
default to `[0, 1]`.
