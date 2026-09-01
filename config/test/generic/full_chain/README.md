# Full-chain metric reporting

`analyzers_240805.yaml` configures the full-chain analyzers to emit segmentation,
PPN, clustering, and matched node-prediction records. Include it in the
inference configuration. It enables fragment construction because fragment
primary classification depends on constructed fragment objects.

`report_240805.yaml` reduces those CSV files after every inference shard has
completed. The reporter is a separate CPU-only process; it does not initialize
the SPINE `Driver`, Torch, or LArCV.

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
values, and plots. Each plot is written as soon as its recipe finishes. If a
later recipe fails, the incrementally updated `summary.json` still describes
every completed plot. If a checkpoint path is configured without a checksum,
the reporter calculates its SHA-256 digest while streaming the file.

The scheduler should submit this command as one small CPU job with an `afterok`
dependency on every inference array or chunk job that contributes CSVs. The
reporter does not contain scheduler-specific logic.

PPN distances are streamed in configurable chunks. `distance_scale` can be used
when the analyzer distance unit needs conversion before thresholds and
histograms are evaluated. Clustering metric ranges are configurable through a
`metric_ranges` mapping; ARI defaults to `[-1, 1]`, while efficiency and purity
default to `[0, 1]`.

Node tasks consume matched `save_truth_*` rows. Nested `quality_cuts` support
`all`/`any` composition and column predicates, keeping overlap, minimum-size,
particle-shape, primariness, and ancestry selections explicit in the report
configuration. Classification tasks produce confusion matrices. Orientation
tasks summarize the cosine between truth and reconstructed start directions.
