Data Model
==========

SPINE passes typed data products between pipeline stages. Detector records,
reconstruction objects, truth objects, tensors, indexes, and batch containers
all live under :mod:`spine.data`; computational packages consume them but do
not own duplicate representations.

Object Hierarchy
----------------

Constructed outputs form three levels:

- fragments are groups of detector points produced by fragmentation;
- particles combine fragments and carry identity, direction, and kinematics;
- interactions group particles that share an interaction hypothesis.

Each level has reconstructed and truth variants. Reconstructed objects carry
predictions and derived quantities. Truth objects add simulation identifiers,
ancestry, deposited-energy information, and truth-to-reconstruction matching
state. See the :doc:`api/data` reference for the complete set of products.

Reading Dataclass References
----------------------------

SPINE data objects use dataclasses with multiple inheritance. A conventional
autodoc page flattens every member into one alphabetical list, obscuring which
values are stored, computed, or inherited. The data API therefore generates a
structured reference directly from the class definitions and docstrings.

Each dataclass page separates:

``Stored fields``
   Constructor and serialized state, grouped by the class that declares the
   effective annotation. Each row includes the annotation, default, field
   description, and SPINE metadata such as units, index semantics, or array
   length.

``Computed properties``
   Values derived from stored state. These are not independent constructor
   inputs even if a writer can persist selected derived values.

``Methods``
   Public operations, grouped by their declaring class with full signatures
   and method documentation.

The method-resolution order at the top of each page explains how overlapping
definitions are selected. For example, the
:class:`spine.data.out.TruthParticle` page keeps particle-specific state
visually separate from fields inherited through its output, positional, and
truth bases. This layout is introspection-driven: adding or moving a documented
dataclass field updates the reference without maintaining a hard-coded member
list.

Units And Coordinates
---------------------

Many point-like fields may be expressed in detector coordinates or physical
centimeters. Consult the field metadata shown in the API reference and the
object's ``units`` state before combining coordinates from different products.
Geometry transformations belong to :mod:`spine.geo`; data classes describe
the state but do not silently perform detector-specific conversion.

Event And Batch Products
------------------------

Event products under :mod:`spine.data.product` pair arrays with schema
information needed at I/O and model boundaries. Batch products under
``spine.data.product.batch`` add batch counts or offsets while preserving the
event-level meaning. Unwrapping restores event products before construction,
post-processing, and analysis consume them.
