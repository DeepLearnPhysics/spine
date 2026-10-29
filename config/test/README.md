# Full-chain metric reporting

`analyzers.yaml` configures the full-chain analyzers to emit segmentation,
PPN, clustering, and matched node-prediction records. Include it in the
inference configuration **after** a full-chain base with an ordered `post.stages`
list containing `match` (for example `config/full_chain/full_chain_regression.yaml`).
It updates `match` by name and inserts `ppn` after it, preserving the other
processors and their execution order. Its `ana.stages` list also executes in
written order. It enables fragment construction because fragment
primary classification depends on constructed fragment objects.

`report.yaml` reduces those CSV files after every inference shard has
completed. The reporter is a separate CPU-only process; it does not initialize
the SPINE `Driver`, Torch, or LArCV.

Run it with:

```bash
spine-report \
  --config config/test/report.yaml \
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
histograms are evaluated. Its `classes` selection controls the separate
per-class curves, while `overall_classes` independently controls the aggregate
efficiency/purity population. The generic configuration reports delta rays as
a separate class but conventionally excludes them from the overall curves.
Clustering metric ranges are configurable through a
`metric_ranges` mapping; ARI defaults to `[-1, 1]`, while efficiency and purity
default to `[0, 1]`.

Class labels come from `spine.constants`. A `classes` list restricts a recipe
to selected canonical IDs or names. A `class_mapping` map combines source
classes under new report labels, for example mapping every semantic category
except ghost to `Non-ghost`.

```yaml
# Restrict a shape-based recipe without repeating display labels.
classes: [shower, track]

# Or aggregate canonical classes under report-specific labels.
class_mapping:
  Non-ghost: [shower, track, michel, delta, low_energy]
  Ghost: [ghost]
```

The generic report demonstrates both forms. Its standard segmentation matrix
is explicitly restricted to shower through low-energy scatter, the complete
set present in the generic dataset. A separate `segmentation_shower_track`
metric maps shower, Michel, delta, and low-energy scatter to `Shower`, while
keeping track as `Track`. Because each is an independent recipe, both records
and both confusion plots are retained.

Node tasks consume matched `save_truth_*` rows. Nested `quality_cuts` support
`all`/`any`/`not` composition and column predicates, keeping overlap,
minimum-size, neutrino origin, particle shape, primariness, and ancestry
selections explicit. Classification tasks produce confusion matrices.
Orientation tasks summarize the cosine between truth and reconstructed start
directions.

The `shower_primary_eval` analyzer emits one row per modeled reconstructed
shower-network node (shower, Michel and delta by default), with `target`,
`prediction`, `valid`, `shape`, `fragment_id` and `target_definition`. The
`shower_fragment_primary` report uses these node-aligned rows and explicitly
requires `valid: true`. No best-match truth-object join is needed.

Configure its target policy identically to `NodeShowerPrimaryLoss`: `use_closest`,
`high_purity`, `use_group_pred`, `min_iou`, `min_purity`, `min_efficiency` and
`match_target`. The generic defaults mirror the generic full-chain objective.
For a closest-fragment objective with predicted-group purity, use:

```yaml
- name: shower_primary_eval
  use_closest: true
  high_purity: true
  use_group_pred: true
```

This needs `clust_label_adapt` on the reconstructed point domain, `coord_label`
when closest selection is enabled, and the fragment clusters, shapes, node
logits and (for predicted purity) group predictions. Preserve these products
when running inference, and provide the particle-coordinate parser if it is
not already configured. Set `shapes` to the integer shape IDs modeled by the
shower network if they differ from the defaults. Fragments with unmodeled
`-inf` logit sentinels do not enter targets or purity masks.

Training and reporting share the same closest-fragment and purity kernels;
overlap cuts are also applied through the shared quality policy. Only the
fragment of the *true primary particle* closest to its shower start is selected,
not the closest arbitrary fragment in the shower. The raw-logit argmax used by
training is reported, rather than a downstream group-constrained assignment.
The former matched-object comparison is retained as
`shower_fragment_group_primary`; it measures a different target and population.
Old CSVs can still use that task, but must rerun inference for the new records.

Full-chain truth fragments are constructed from labels adapted to the semantic
prediction. Their populated voxel attributes are therefore `index_adapt` and
`size_adapt`; the original `index` is empty. The analyzer configuration uses
adapted indexes for fragment matching and clustering, and the fragment-primary
quality cut uses the saved `truth_size_adapt` column.

Semantic recall denominators
---------------------------

`segment_confusion` retains existing `recall` and `support` values. The explicit
alias `recall_conditional` uses the displayed class-restricted matrix: true
points predicted outside that subset are omitted from its denominator.
`support_overall` instead counts the selected truth class across **all** raw
prediction rows, and `recall_overall` divides the correctly classified count by
that original support. Thus predictions removed by deghosting reduce overall
semantic recall without changing the meaning of conditional segmentation
recall. Class mappings pool truth support consistently before calculating the
ratios. Existing `segment_eval_summary.csv` shards suffice to regenerate these
fields if they include ghost prediction rows (`segment_eval.ghost: true`);
a shard recorded after those rows were discarded cannot recover them.
When the two supports differ, an additional `*_recall.png`/PDF figure compares
conditional and overall recall alongside the unchanged confusion plot.

These additions do not reinterpret adapted-space clustering efficiency as
original-truth efficiency; those remain separate metric definitions.
