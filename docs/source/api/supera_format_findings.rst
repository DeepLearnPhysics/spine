pysupera 4.1 integration findings
=================================

This report is prepared for the pysupera authors from the SPINE PR #191
integration. It distinguishes an incorrect documented contract from intentional
loss of source detail and requests for a more explicit interchange schema.
The author responses below supersede the original open questions.

.. note::

   This is the historical investigation of the retired direct pysupera adapter.
   The measurements remain useful, but descriptions of SPINE's deposit-based
   energy attribution and endpoint fallbacks below do not describe the current
   :ref:`PIMM integration <pimm-input>`.

Author clarification and integration direction
----------------------------------------------

The authors confirmed that the ``io_v3.py`` index docstring is outdated:
stored IDs are pre-filter in-memory particle indexes. They intend to change
these to stored row indexes. Own-particle point ranges describe deposits before
merging and are not expected to cover contributions from omitted particles.
Only LE contributions may merge across interaction boundaries. Self-parenting
is a supported primary criterion; ``geant4_parent_trackid == -1`` is the
preferred direct test. Fragment endpoint information is being added upstream.

Crucially, detector training must read the separate JAXTPC hits, not
``points/flat``. The latter contains voxelized EDepSim energy in MeV;
``hit_energy=charge`` describes the hits and does not change that product.
Entry ``k`` of ``hit_labels/volumeV/plane0`` labels entry ``k`` of the matched
JAXTPC pixel hit array, with the source entry given by ``events/jaxtpc_entry``.

SPINE is pivoting to a ``pimm`` provider using ``pimm-data`` for detector input
and pysupera for truth. The measurements below remain useful observations of
the original sample, but are not evidence of detector-charge training support.

Reference production
--------------------

* File: ``supera_pixel_j000000-000004.h5``, 1,000 events, format ``4.1.0``.
* Embedded software identity: ``drinkingkazu/pysupera`` commit
  ``dab92d8fc850c09d0ca85cff16c6f36c28cf6d4b``, clean working tree.
* Producer: ``doraemon_prod.stages.supera``, EDepSim plus JAXTPC pixel input.
* Configuration: 3 mm stored voxels, ``hit_energy=charge``,
  ``hit_x_from=nominal``, ``kinematics_steps=visible``.

The source reference is `the pinned upstream tree
<https://github.com/drinkingkazu/pysupera/tree/dab92d8fc850c09d0ca85cff16c6f36c28cf6d4b>`_.
Measurements below use event 0, reading only that event's particle, interaction
and point slices. No hit-label payload was loaded.

1. Sparse IDs contradict the documented row-index contract
----------------------------------------------------------

**Confirmed documentation/contract mismatch.** ``io_v3.py`` describes particle
IDs as event-local indexes into ``events/offsets[ev] + id``. The file instead
has 5,896 particle rows, with source IDs spanning 0 through 14,152.
``layout.py`` writes the retained original particle IDs and redirects parent
links in that same source-ID namespace. Sparse IDs are not themselves a format
error; interpreting them as rows is.

SPINE originally normalized some relationships but retained source IDs in its
cluster ``particle`` label. Truth building then failed with
``Invalid particle ID found in fragment labels.`` That mixed-namespace adapter
bug is fixed by mapping all internal relationships to particle-table rows and
preserving source IDs in a separate provenance product.

Requested upstream change:

* Specify the namespace of every ID column, including ``frag_parent_id``,
  ``inst_parent_id`` and hit/group labels. Publish the negative sentinel.
* Either write normalized row indexes plus a separate ``source_id``, or retain
  sparse IDs and correct the reader documentation and examples to require a
  source-ID-to-row map. Do not change the meaning silently within version 4.1.
* Validate uniqueness and nonnegative-ID referential integrity per event.

2. Own-particle ranges do not partition the stored deposits
-----------------------------------------------------------

**Observed loss of per-source attribution, consistent with selective row
retention in** ``layout.py``. The event has 28,789 stored point rows. The union
of the stored particles' ``pc_start:pc_end`` ranges covers only 15,334 of them,
without overlaps. The remaining 13,455 points contribute 4,686.0472 MeV of the
11,938.2904 MeV total. Summing only stored own-particle ranges gives
7,252.2432 MeV.

Fragment and instance ranges do partition all point rows. ``layout.py`` retains
visible representatives and ancestry, not every contributing source particle.
The missing own-particle coverage therefore does **not** establish that the
file has lost deposit energy or that its fragment ranges are corrupt. It means
that complete per-source-particle energy cannot be recovered by summing those
stored own-particle ranges.

SPINE accounts for all energy by integrating each fragment range exactly once
and assigning that contribution to its representative. Truth groups sum those
fragment contributions. This is fragment-attributed energy, not an assertion
that the representative's own Geant4 track deposited all of it. Overlap-winner
voxel sums and source-fragment energies are separate quantities.

Requested upstream change:

* Explicitly document that own-particle ranges may leave gaps, while fragment
  and instance ranges are complete partitions.
* Store an explicit per-fragment/per-instance ``energy_deposit`` with units and
  attribution semantics, or guarantee its derivation from the ranges.
* If complete source-particle truth is intended, retain the omitted contributing
  rows or add a deposit-to-original-track association. SPINE cannot reconstruct
  that information from fragment totals.

3. Fragment endpoints are absent; deriving them is approximate
--------------------------------------------------------------

**Missing information for fragment-level supervision, not a malformed array.**
There are 1,046 instance representatives, of which 815 have finite stored first
positions. No non-instance row has a finite ``first_pos``. After SPINE's common
grid and overlap selection, 270 non-LE fragments remain visible; 193 of these
lack stored first positions.

``kinematics.py`` explicitly produces ``first_*``/``last_*`` only for instances.
They describe counted Geant4 step starts/ends, preferentially from particles
with non-LE membership. Showers intentionally have no last endpoint.
``start_*``/``end_*`` instead describe a track's creation/destruction and are
not interchangeable with the observed trajectory endpoints.

SPINE needs one coordinate-label row per stored particle for primary matching
and orientation supervision. It retains stored collected-step endpoints when
available. Missing endpoints are derived from earliest/latest deposits in the
fragment, or the instance's non-LE range when applicable. These positions are
voxelized deposit locations: they cannot recover the exact original step
boundary or sub-voxel trajectory. Rows without usable deposits remain NaN;
non-track end labels repeat their start labels.

Requested upstream change:

* Provide collected-step endpoint coordinates/times for fragment representatives
  as well as instances, with the same LE and visibility policy.
* Define validity independently for position, time and momentum. Record whether
  a coordinate is a true step boundary, deposit midpoint or voxel center.
* State the intended policy for equal-time deposits and split tracks. An
  explicitly versioned endpoint product would remove SPINE's approximation.

4. Source interaction and host interaction can differ
-----------------------------------------------------

**Two legitimate meanings currently share indirect representation.**
``layout.py`` explicitly permits particles' deposits to be absorbed into groups
from another interaction. Particle rows retain their own interaction; points
are ordered by their group's interaction.

In event 0, 19 fragment rows have an ``interaction_id`` different from their
host instance. After overlap selection, 26 winning voxels have this mismatch.
Using the fragment row's own interaction for grouping supervision is therefore
incorrect for those voxels. SPINE uses the host instance's interaction for
cluster supervision and reconstructed truth-group membership. The stored
interaction vertex supplies a common vertex within that group.

Requested upstream change:

* Expose distinct ``source_interaction_id`` and ``instance_interaction_id``
  concepts, or document the required ``frag_inst_id`` join next to the field.
* State explicitly which interaction should supervise deposit/fragment/instance
  grouping, and whether cross-interaction absorption is intended for training.
* Retain the original generator ``vertex_id`` and document its relation to the
  event-local interaction table.

5. Stored roots and true generator primaries need distinct definitions
----------------------------------------------------------------------

**Semantic ambiguity outside the retained genealogy.** ``parent_id`` points to
the nearest stored ancestor and becomes self when none is available;
``ancestor_id`` is the root of that redirected graph. Event 0 has 61 such roots
and 17 interactions. This graph is useful and walkable, but a stored root is
not, by definition alone, proof of generator-primary status. Real direct-parent
track/PDG columns are also available and should retain their separate meaning.

SPINE preserves the stored genealogy, uses a host's self-parent status for
interaction-primary supervision, and propagates that label to its fragments.
Missing parents remain unknown. This convention should be reviewed by the
producer before treating it as an independent physics-primary definition.

Requested upstream change:

* Define and, preferably, store ``interaction_primary`` explicitly, including
  the treatment of prompt decays and missing/unwritten ancestors.
* Distinguish redirected stored-parent links from original Geant4 parents.
  Document self-root versus negative/missing sentinel behavior.
* Preserve full generator/neutrino truth if consumers are expected to recover
  it. An interaction position alone cannot supply neutrino identity or missing
  process information.

6. Truth deposits, measured charge and association domains
----------------------------------------------------------

**Schema clarity request, not a discovered charge-conversion bug.** Despite
``hit_energy=charge``, the file's ``points/flat`` is truth-deposit dE in MeV,
not measured pixel charge. The measured hit payload remains in the separate
JAXTPC file. A model trained on this product is learning from truth-deposit
energy, not detector charge. Its learned weights are not thereby validated for
measured-charge inference.

``groups`` has 54,863,490 rows over the production, versus 31,871,012 point
rows. The group domain is JAXTPC deposit groups concatenated over volumes, not
point rows. Hit labels occupy yet another, very large domain. Their offset
relationships must be explicit; equal-length or row-alignment assumptions are
unsafe.

Requested upstream change:

* Attach units, value kind (truth energy versus measured charge), coordinate
  convention and row-domain metadata to each product, not only production
  configuration attributes.
* If standalone detector-input training is intended, include the measured hit
  payload or a stable source-file identity/entry/volume mapping, plus an
  explicitly defined many-to-many hit-to-truth association.
* Publish a schema with per-product offsets, cardinality checks and examples
  that read event slices rather than whole hit-label arrays.

Minimal reproduction
--------------------

This checks the ID/range findings without loading whole-file point or hit data:

.. code-block:: python

   import h5py
   import hdf5plugin  # Register the file's compression filters.
   import numpy as np

   with h5py.File("supera_pixel_j000000-000004.h5", "r") as f:
       p0, p1 = f["events/offsets"][:2]
       x0, x1 = f["points/offsets"][:2]
       p = {key: value[p0:p1] for key, value in f["particles"].items()}
       points = f["points/flat"][x0:x1]
       columns = list(f["points/columns"].asstr()[:])
       cover = np.zeros(len(points), dtype=np.int32)
       for start, stop in zip(p["pc_start"], p["pc_end"]):
           if start >= 0:
               cover[start - x0:stop - x0] += 1
       energy = points[:, columns.index("dE")]
       print("particle rows, max ID:", len(p["id"]), p["id"].max())
       print("own-range coverage:", np.unique(cover, return_counts=True))
       print("total MeV:", energy.sum(dtype=np.float64))
       print("unattributed MeV:", energy[cover == 0].sum(dtype=np.float64))

Suggested upstream acceptance checks
------------------------------------

Before freezing the format, validate a fixture containing sparse source IDs,
absorbed/unwritten contributors, cross-interaction absorption, a split track,
a shower, missing endpoints and an empty event. Check ID namespaces, complete
fragment/instance point coverage, energy conservation, endpoint validity and
source-versus-host interaction semantics. Round-trip the fixture through the
upstream reader and a small independent consumer. Changing these contracts
should increment the format version.
