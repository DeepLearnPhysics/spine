# Documentation

[![Documentation Status](https://readthedocs.org/projects/spine/badge/?version=latest)](https://spine.readthedocs.io/latest/)

We use Sphinx to generate the documentation, and Read the Docs to host it at https://spine.readthedocs.io/latest/.
CI and Read the Docs treat every Sphinx warning as a build failure.

## API documentation

SPINE uses autosummary for package, module, function, and conventional class
references. Public dataclasses use the custom `spine-dataclass` directive
through `docs/source/_templates/dataclass.rst`. It derives the reference from
the dataclass definitions and NumPy-style docstrings, and presents:

1. stored fields grouped by their declaring class;
2. computed properties in a separate section;
3. public methods with signatures and documentation;
4. types, defaults, units, and other SPINE field metadata.

This avoids flattening complex objects such as `TruthParticle` into one
alphabetical member list. Do not copy inherited fields into a child class's
docstring: document each field where it is declared.

### Adding API entries

Add a public symbol to the appropriate file under `docs/source/api/`. Modules,
functions, and conventional classes use standard autosummary:

```rst
.. autosummary::
   :toctree: generated

   module_name
```

For a dataclass, select the structured template:

```rst
.. autosummary::
   :toctree: generated
   :template: dataclass.rst

   out.TruthParticle
```

## Writing docstrings

Use NumPy style. See [Napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html) and [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) style guides.

### Documenting a class with attributes

```python
class MyClass:
    """Short description.
    
    Longer description explaining what this class does.
    
    Attributes
    ----------
    param1 : int
        Description of param1
    param2 : str, optional
        Description of param2
    """
```

### Documenting a generic function
```python
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

```python
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

### Quick build (recommended)

```bash
cd docs/
./build_docs.sh
```

This removes generated autosummary sources, cleans the HTML output, and runs a
warning-strict build matching CI.

### Manual build

If you would like to build it yourself on your local computer:

```bash
cd docs/
pip install -r requirements.txt
rm -rf source/api/generated
make clean
make html SPHINXOPTS="-W --keep-going"
```

Then open the file `docs/build/html/index.html` in your favorite browser:

```bash
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

### On Read the Docs
The configuration for this build is in `../.readthedocs.yaml`.

`requirements_rtd.txt` includes the same documentation dependency set as the
local build.

ReadTheDocs automatically builds on every push to main.
