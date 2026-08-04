## Tests

The default test suite covers dependency-light SPINE functionality. Tests that
need the complete model runtime are marked `model` and run in CI inside the
published SPINE image.

### Model contracts

The maintained model configurations are listed in
`test/test_model/cases.py`. Every registered model must have at least one
checked-in configuration. Canonical standalone configurations live under a
model-specific directory, with `<model>_train.yaml` and `<model>_test.yaml`
representing training and inference respectively. UResNet is the initial
prototype for this convention. Canonical training configurations express the
run duration with `base.epochs` and checkpoint cadence with
`base.train.save_epoch`, keeping both independent of dataset size.

Model testing has three levels:

1. Configuration and registry checks run without PyTorch.
2. Construction tests instantiate the network and loss for every maintained
   configuration.
3. Execution tests run one loader, forward, loss, backward, and optimizer
   iteration for each standalone configuration.

UResNet and the full reconstruction chain have deterministic LArCV regression
tests. Each test runs the maintained inference configuration twice, checks
same-machine repeatability, and compares compact output summaries with a
checked-in reference.

Run the model contracts in a full SPINE environment with:

```bash
pytest test/test_model
```

### External test data

The LArCV and HDF5 fixtures download the small files configured in
`test/pytest.ini` once per test session.

### Useful selections

```bash
pytest -m "not slow"
pytest -m model
pytest -m gpu
```
