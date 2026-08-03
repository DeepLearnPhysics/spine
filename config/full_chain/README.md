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

External providers can be registered with `register_provider`, or referenced
as `package.module:PROVIDER_SPEC`. A provider may publish several capabilities
at once—for example, semantic predictions and fragments—without any change to
`FullChain`.
