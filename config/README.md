# Example configurations

These configurations use `provider` to select implementations, including models,
IO components, optimizers, and parsers. Parameters may be inline or nested under
`config`; use one form per descriptor. Mapping keys such as dataset outputs and
prediction heads remain their identifiers.

Managers that execute a sequence use `stages`: each entry requires a unique
`name`, and its `provider` defaults to that name. Entries run in written order;
`priority` belongs only to the legacy mapping format.

```yaml
post:
  stages:
    - name: direction
    - name: calorimetry
      provider: calo_ke
      config:
        scaling: 1.0
```

Includes resolve in order. Use named-list edits (`override: post.stages~`) to
modify an existing ordered pipeline without replacing it. See
[test/README.md](test/README.md) for the analyzer fragment's base requirements,
and the [configuration reference](../src/spine/config/README.md) for the full
syntax and migration rules.
