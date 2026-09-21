# Full-chain configuration

The full chain is an ordered list of providers. Each provider consumes and
publishes named reconstruction products; the orchestrator does not contain a
hard-coded model order.

## Maintained configurations

The directory contains three related entry points:

- `full_chain_train.yaml` is the canonical end-to-end training example;
- `full_chain_test.yaml` is the canonical model-only inference/evaluation
  example;
- `full_chain_regression.yaml` is the checkpoint-pinned integration job used
  to validate model, reconstruction, and post-processing output together.

The train and test examples inherit the common model definition from the
regression configuration, but remove builders, post-processors, the writer,
and post-only input products. Their resolved configurations therefore contain
only `base`, `io`, and `model` sections while avoiding three independent copies
of the full model definition.

Each provider adapter declares four parts of its contract:

- `requires`: canonical products that must already exist;
- `optional`: canonical products used when available;
- `provides`: canonical products made available to later stages;
- `replaces`: existing products that it may intentionally supersede.

The plan is dependency-checked at construction time, and required inputs are
checked again at execution time. Public model outputs are kept separate from
canonical inter-stage products, so adding a diagnostic output cannot silently
change downstream reconstruction.

```yaml
model:
  name: full_chain
  modules:
    chain:
      stages:
      - name: segmentation
        provider: segmentation
        uses: uresnet_ppn
        loss: uresnet_ppn_loss
        config:
          mode: uresnet
          point_proposal: ppn
      - name: fragmentation
        provider: fragmentation
        uses: graph_spice
        loss: graph_spice_loss
        config:
          mode: graph_spice
```

## Deghosted charge diagnostics

Reconstructed charge redistribution can optionally retain its per-voxel
inputs. Enable `store_charge_info` on the deghost provider together with
`average` or `collection` charge rescaling:

```yaml
      - name: deghosting
        provider: deghost
        config:
          mode: uresnet
          charge_rescaling: average
          store_charge_info: true
```

This publishes `charge_per_plane` and `charge_multiplicity`, each with three
columns and the same post-deghost row partition as `data_adapt`. Together they
record the plane charge and number of retained space points sharing each hit,
so the redistributed contribution from an active plane is
`charge_per_plane / charge_multiplicity`. These potentially large diagnostics
are absent by default. In the historical mode-matrix schema,
`store_charge_info` belongs beside `charge_rescaling` in `modules.chain`.

`uses` lists sibling module blocks injected into the provider configuration.
`loss` is either one sibling loss block or a mapping of provider-owned task
names to blocks. The historical `chain` mode matrix remains accepted and is
translated into this representation during construction.

## Calibration placement

In the ordered schema, calibration is an ordinary provider. Its position in
the `stages` list determines when it runs, so its configuration does not need
a `stage` option:

```yaml
    chain:
      stages:
      - name: calibration
        provider: calibration
        uses: calibration
        config:
          mode: apply
      - name: segmentation
        provider: segmentation
        uses: uresnet_ppn
        config:
          mode: uresnet
```

The historical mode-matrix schema is unordered. For that schema only,
`modules.calibration.stage` remains required so the compatibility translator
knows which generated stage calibration must precede. The translator consumes
that option; it is not part of the calibration provider interface.

## GrapPA graph materialization

Cache jobs can stop at GrapPA's deterministic graph boundary instead of
running message passing, prediction heads, grouping, and object construction.
Fragment-node materialization supports the shower, track, and joint-particle
paths:

```yaml
    chain:
      stages:
      - name: fragmentation
        provider: fragmentation
        uses: graph_spice
        config:
          mode: graph_spice
      - name: fragment_graph
        provider: fragment_graph
        uses: [grappa_shower, grappa_track]
        loss:
          shower: grappa_shower_loss
          track: grappa_track_loss
```

The stage intrinsically publishes each path's edge index, node and edge
features, and configured static target/validity pairs. GrapPA checkpoints and
the historical ``return_features`` and ``return_targets`` overrides are not
needed. If grouped or class-selected GrapPA node dropout is configured, the
stage also materializes its truth-derived group IDs and eligibility mask from
``clust_label``. Cached training then samples a fresh dropout selection on
every iteration.

Interaction GrapPA uses particles as graph nodes. A particle-node cache job
may either receive the canonical particle products as declared chain inputs or
run an upstream provider which constructs them:

```yaml
    chain:
      inputs: [particle_clusts, particle_shapes, particle_primaries]
      stages:
      - name: particle_graph
        provider: particle_graph
        uses: grappa_inter
        loss: grappa_inter_loss
```

Both providers cache only iteration-independent supervision. Dynamic forest
selection and predicted-group purity are rebuilt from the current network
outputs during cached training.

## Particle image tasks

An image encoder can classify or regress reconstructed particle clusters after
particle aggregation. It is independent of interaction GrapPA and can replace
individual GrapPA node heads while leaving interaction aggregation enabled.

```yaml
    chain:
      stages:
      # Segmentation, fragmentation and particle aggregation precede this.
      - name: particle_tasks
        provider: particle_image
        uses: image_particle
        loss: image_particle_loss
      - name: interaction_aggregation
        provider: interaction_aggregation
        uses: grappa_inter
        loss: grappa_inter_loss
        config:
          mode: grappa
          task_modes:
            type: image
            primary: image
            orient: grappa

    image_particle:
      objects:
        source: explicit
      encoder:
        name: cnn
        num_input: 1
        spatial_size: 768
        filters: 32
        depth: 5
        reps: 2
      heads:
        pid: 5
        primary: 2

    image_particle_loss:
      pid:
        name: class
        label: clust_label
        target: pid
        loss: ce
      primary:
        name: class
        label: clust_label
        target: interaction_primary
        loss: ce
```

When an image provider owns a task, remove that task's named node loss from
`grappa_inter_loss`. The chain rejects duplicate GrapPA supervision during
construction because the corresponding GrapPA logits are intentionally not
published.

The `pid` image head is published as `particle_node_type_pred`, so existing
particle builders consume it exactly as they consume GrapPA's type prediction.
Likewise, `orientation` maps to `particle_node_orient_pred`. Arbitrary extra
heads such as energy regression are published as
`particle_node_<head>_pred`.

## Interaction vertexing

Interaction vertexing is a reduction stage after interaction aggregation. It
publishes one `interaction_vertices` row and one
`interaction_vertex_scores` value for every `interaction_clusts` entry. The
interaction builder consumes the vertex product when available and otherwise
retains its unset (`NaN`) vertex.

A vertex PPN supplies voxel-aligned proposals through the segmentation stage:

```yaml
    chain:
      stages:
      # Segmentation must use a uresnet_ppn block with a `vertex` head.
      - name: segmentation
        provider: segmentation
        uses: uresnet_ppn
        config:
          mode: uresnet
          point_proposal: ppn
      # Fragmentation and both aggregation stages precede this reducer.
      - name: interaction_vertexing
        provider: interaction_vertexing
        config:
          mode: ppn
          score_threshold: 0.5
          pool_radius: 1.999
          pool_score_fn: max
```

Interaction GrapPA can instead expose a named five-output vertex head: two
primary logits and three position values. The reducer chooses the particle
with the largest primary probability in each predicted interaction.

```yaml
    chain:
      stages:
      - name: interaction_aggregation
        provider: interaction_aggregation
        uses: grappa_inter
        loss: grappa_inter_loss
        config:
          mode: grappa
      - name: interaction_vertexing
        provider: interaction_vertexing
        config:
          mode: grappa
          normalize_positions: false
          use_anchor_points: false

    grappa_inter:
      gnn_model:
        node_pred:
          vertex: 5

    grappa_inter_loss:
      node_loss:
        vertex:
          name: vertex
          normalize_positions: false
          use_anchor_points: false
```

Anchor mode additionally requires an endpoint-producing GrapPA node encoder,
such as the geometric encoder with `add_points: true`. The legacy schema uses
`chain.vertexing: ppn` or `chain.vertexing: grappa`; reducer options belong in
the sibling `interaction_vertexing` module block. When a learned vertex is
enabled, omit the geometric `post.vertex` processor unless intentionally
replacing the learned result.

External providers can be registered with `register_provider`, or referenced
as `package.module:PROVIDER_SPEC`. A provider may publish several capabilities
at once—for example, semantic predictions and fragments—without any change to
`FullChain`.
