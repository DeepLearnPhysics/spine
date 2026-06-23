# Documentation

[![Documentation Status](https://readthedocs.org/projects/spine/badge/?version=latest)](https://spine.readthedocs.io/latest/)

We use Sphinx to generate the documentation, and ReadTheDocs.org to host it at https://spine.readthedocs.io/latest/.
The online documentation gets built and updated automatically every time the main branch changes.

## Automatic Class Documentation

SPINE uses **automatic docstring inheritance** and **Sphinx autosummary** to provide comprehensive class documentation. When you update a class, the documentation automatically:

1. **Merges inherited attributes** from parent classes (via `__init_subclass__`)
2. **Generates individual class pages** with complete docstrings
3. **Updates on every git push** via ReadTheDocs

No manual intervention needed! Just write good docstrings in your classes and they'll appear beautifully formatted in the docs.

### How It Works

Classes with inheritance (like `RecoFragment` inheriting from `OutBase`, `FragmentBase`, `RecoBase`) automatically merge their parent attributes into their docstrings. When Sphinx builds the docs, it sees the fully merged docstring and generates complete documentation.

**Example:**
```python
class RecoFragment(RecoBase, FragmentBase, OutBase):
    """Reconstructed fragment.
    
    Attributes
    ----------
    shape : int
        Predicted shape (from RecoBase)
    """
```

The documentation will show **all** attributes: those from `RecoBase`, `FragmentBase`, `OutBase`, plus `shape`.

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

This script will clean, build, and show you where to open the result.

### Manual build

If you would like to build it yourself on your local computer:

```bash
cd docs/
pip install -r requirements.txt
make clean  # Clean previous build
make html   # Build HTML
```

Then open the file `docs/build/html/index.html` in your favorite browser:

```bash
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

### Adding new classes to documentation

To add new classes to the API documentation:

1. **Add the class to the appropriate .rst file** in `docs/source/api/`
2. **Use autosummary and autoclass directives**:

```rst
.. autosummary::
   :toctree: generated/
   :nosignatures:

   spine.module.NewClass

.. autoclass:: spine.module.NewClass
   :members:
   :inherited-members:
   :show-inheritance:
```

3. **That's it!** The documentation will automatically include all merged docstrings.

### On ReadTheDocs.org
The configuration for this build is in `../.readthedocs.yaml`.

The dependencies used by the build are in `requirements_rtd.txt`.

ReadTheDocs automatically builds on every push to main.
