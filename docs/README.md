# Documentation

[![Documentation Status](https://readthedocs.org/projects/spine-ml/badge/?version=latest)](https://spine-ml.readthedocs.io/en/latest/?badge=latest)

We use Sphinx to generate the documentation, and ReadTheDocs.org to host it at https://spine-ml.readthedocs.io/en/latest/.
The online documentation gets built and updated automatically every time the main branch changes.

## Writing docstrings
Use NumPy style. See [Napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html) and [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) style guides.

### Documenting a generic function
```
def func(arg1, arg2):
    """Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value
    """
    return True
```

### Documenting a ML model
For an ML model, please try to document `Configuration` (YAML Configuration options) and `Output` (keywords in the output dictionary) sections:

```
class MyNetwork(torch.nn.Module):
    """
    Awesome network!

    Configuration
    -------------
    param1: int
        Description

    Output
    ------
    coordinates: int
        The voxel coordinates
    """
```

## Building the documentation
### Locally
If you would like to build it yourself on your local computer:

```
$ cd docs/
$ pip install -r requirements.txt
$ sphinx-apidoc -f -M -e -T -o ./source ../mlreco/ ../mlreco/models/arxiv/
$ sphinx-apidoc -f -M -e -T -o ./source ../analysis/
$ make html
```

Note: `sphinx-apidoc` generates automatically a .rst file for each Python file
it detected (recursively). It needs a `__init__.py` file in a folder for
it to be recognized as a Python package.

Then open the file `docs/_build/html/index.html` in your favorite browser.

If you make changes to the documentation only, just run `make html` every time
to re-build it. If however you change the file hierarchy, you may want to re-run
`sphinx-apidoc`.

### On ReadTheDocs.org
The configuration for this build is in `../.readthedocs.yaml`.

The dependencies used by the build are in `requirements_rtd.txt`.
