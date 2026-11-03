Shared UResNet Reconstruction Model
==================================

``model.provider: uresnet_multi`` selects ``UResNetMulti`` and
``UResNetMultiLoss`` from ``spine.model.uresnet.multi``. One UResNet supplies
semantic segmentation, particle-point proposals and Graph-SPICE fragmentation.
Optional vertex proposals retain the UResNet-PPN configuration contract.

The implementation separates the network orchestration in ``multi/model.py``,
joint objectives in ``multi/loss.py``, and fragment classification/pooling in
``multi/semantics.py``. Existing standalone UResNet-PPN and Graph-SPICE models
remain available with their original parameter paths.

Architectural options
---------------------

The ``semantic_coupling`` setting selects the relationship between voxel
semantics and fragmentation:

.. list-table:: Supported coupling modes
   :header-rows: 1
   :widths: 18 28 29 25

   * - Mode
     - Candidate graph
     - Edge prediction
     - Default final semantics
   * - ``hard``
     - Separate graph per predicted class
     - Learned voxel features
     - Class used for routing
   * - ``independent``
     - One graph per event over all input voxels
     - Learned voxel features; no semantic predictions
     - Mean voxel probabilities per fragment
   * - ``fragment_first``
     - One graph per event over all input voxels
     - Learned voxel features; no semantic predictions
     - Learned classifier on pooled fragment features
   * - ``soft``
     - One graph per event over all input voxels
     - Learned features plus semantic probabilities and relative position
     - Mean voxel probabilities per fragment

All four options use the same backbone, PPN branch, graph neighborhood settings
and canonical full-chain interface. The non-hard modes do not use preliminary
semantic predictions even to select graph nodes. Classes excluded from final
fragment output are filtered only after fragment classification.

``fragment_semantics`` independently selects the final classification policy:

* ``routed`` retains the class used to construct a hard-routed graph. It requires
  ``semantic_coupling: hard``.
* ``pooled`` averages member-voxel class probabilities and assigns the winning
  class to the whole fragment. It introduces no new trainable parameters.
* ``learned`` mean-pools decoder features and applies a learned linear fragment
  classifier. It can refine hard, independent or soft predictions and is required
  by ``fragment_first``.

If omitted, ``fragment_semantics`` uses the defaults shown in the table.
Independent predictions therefore remain independent through edge prediction;
the final pooling step reconciles their outputs to enforce semantic coherence.

Soft semantic conditioning
--------------------------

The soft mode wraps the configured Graph-SPICE kernel with a semantic correction
network. The original kernel consumes its normal raw or projected embeddings.
A small multilayer perceptron receives the sum and absolute difference of the
two probability vectors, plus absolute normalized coordinate differences. Its
output is added to the original edge logit. Cross-class edges remain candidates.

.. code-block:: yaml

   model:
     provider: uresnet_multi
     modules:
       # Include the usual backbone, proposal and graph blocks here.
       semantic_coupling: soft
       fragment_semantics: learned
       semantic_conditioning:
         detach: false
         hidden_channels: 32

``detach: false`` allows the edge objective to train the semantic logits as well
as the shared features. ``detach: true`` treats probabilities as fixed evidence
for the edge objective while retaining ordinary voxel semantic supervision.
These options apply only to soft coupling. The final classifier policy does not
change the candidate graph or the edge feature layout.

Configuration examples
----------------------

Maintained training and inference configurations live under
``config/uresnet/multi``:

* ``uresnet_multi_train.yaml`` and ``uresnet_multi_test.yaml`` implement hard
  routing and serve as the common configuration base.
* ``uresnet_multi_independent_{train,test}.yaml`` uses class-agnostic fragments
  and pooled voxel probabilities.
* ``uresnet_multi_fragment_first_{train,test}.yaml`` adds learned fragment
  classification.
* ``uresnet_multi_soft_{train,test}.yaml`` enables semantic edge conditioning.
  Set ``fragment_semantics: learned`` to add learned final refinement.

Each variant inherits the common data, backbone and objective settings. Set the
input file paths and, for trained inference, the checkpoint path before running.
Changing an architecture that introduces parameters requires a compatible
checkpoint; standalone and joint model checkpoints are not interchangeable.

The module blocks are ``uresnet``, ``ppn`` and ``graph_spice``, with corresponding
``uresnet_loss``, ``ppn_loss`` and ``graph_spice_loss`` blocks. ``vertex``,
``vertex_loss`` and ``proposal_decoder`` are optional.

The ``graph_spice.embedder`` block must not contain ``uresnet`` or ``coord_conv``:
its input features already come from the shared decoder. Set
``uresnet.spatial_size`` for coordinate normalization. The standalone Graph-SPICE
coordinate-input augmentation is not applied automatically to the shared backbone.

Inputs must already be deghosted. Place the existing full-chain deghost provider
upstream instead of setting ``uresnet.ghost: true``.

Outputs and semantic coherence
------------------------------

* ``segmentation`` contains preliminary voxel logits for supervision.
* ``seg_pred_raw`` contains their initial argmax predictions.
* ``seg_pred`` contains final voxel classes consistent with fragment membership.
* Existing ``ppn_*`` outputs retain their meaning.
* ``graph_spice_*`` outputs contain the namespaced graph predictions and indexes.
* ``fragment_clusts`` indexes original input rows and ``fragment_shapes`` contains
  exactly one semantic class per returned fragment.
* Pooled/learned modes also return ``fragment_candidates`` and ``fragment_logits``
  for every candidate, before filtering to the requested final output shapes.

Every candidate member inherits its fragment's class. Candidates classified
outside ``graph_spice.shapes`` are omitted from canonical fragments, but their
member voxels retain that assigned class. Voxels omitted by minimum-size/orphan
handling retain their preliminary prediction. Raw predictions remain available
for diagnostics and are never overwritten by final semantic assignment.

Graph indexes remain in their graph-node domain. Canonical fragments and
candidate fragments are restored to the original input domain, including event
offsets. In class-agnostic mode, internal ``graph_spice_clust_shapes`` and graph
shape fields are partition identifiers, not predicted semantic classes. Use
``fragment_shapes`` for downstream reconstruction.

``graph_spice.make_clusters`` defaults to true. Only hard coupling with routed
semantics can disable clustering during standalone training. Pooled or learned
semantics need predicted components during both training and inference; the
full-chain provider always requires canonical fragment outputs.

Joint supervision and gradient paths
------------------------------------

The default loss sums voxel segmentation, PPN and edge objectives, plus fragment
classification when learned semantics are enabled. ``loss_balancing`` supports
``fixed`` and ``uncertainty`` with the existing PPN leaf names,
``fragmentation``, and optionally ``fragment_semantics``. Empty edge and fragment
objectives contribute no uncertainty regularizer.

Positive edge targets require matching nonnegative original cluster IDs and
matching valid truth semantic classes. Equal negative cluster sentinels do not
create positive edges. Predicted semantic classes never rewrite these targets.

The learned classifier uses the truth-class distribution within each predicted
fragment as a soft categorical target. A mixed fragment is therefore supervised
by its composition rather than an arbitrary majority label. The reported
``fragment_semantics_purity`` measures the average dominant truth-class fraction;
``fragment_semantics_accuracy`` compares predicted classes to the dominant truth
class. All candidates receive supervision, including those currently classified
outside the requested output shapes.

Fragment-first mode retains voxel segmentation as an auxiliary objective. To
study fragment-only semantic supervision, use fixed balancing with
``weights.segmentation: 0.0``. In that experiment, voxels not assigned to a
fragment do not have a directly supervised fallback semantic prediction.

Graph neighborhood construction, connected components and final class assignment
are discrete. Fragment classification gradients train the classifier and pooled
backbone features, but do not differentiate the component membership or edge
thresholding. The edge objective trains connectivity separately. Soft conditioning
adds a differentiable path from edge loss to voxel probabilities when attached.

Uniform final labels guarantee output coherence; they do not prove that a
fragment avoids a true physical boundary. Evaluate fragmentation split/merge
metrics and truth-class purity as well as classification accuracy. The modes
that classify predicted fragments also incur connected-component work during
training, unlike the hard-routing edge-only training configuration.

Full-chain integration
----------------------

Replace separate segmentation and fragmentation stages with the combined
provider in the native ordered-stage configuration:

.. code-block:: yaml

   model:
     provider: full_chain
     modules:
       chain:
         stages:
           - name: reconstruction
             provider: uresnet_multi
             uses: uresnet_multi
             loss: multi_loss
           # Existing fragment materialization and aggregation stages follow.
       uresnet_multi:
         uresnet: {}       # Supply the backbone configuration here.
         ppn: {}
         graph_spice: {}   # Supply kernel/constructor/head configuration here.
         semantic_coupling: soft
         fragment_semantics: learned
       multi_loss:
         uresnet_loss: {}
         ppn_loss: {}
         graph_spice_loss:
           provider: edge
           loss: binary_log_dice_ce

The empty backbone/graph blocks are placeholders for the maintained example's
corresponding blocks. The provider publishes ``seg_pred``, ``ppn_points``,
``fragment_clusts`` and ``fragment_shapes``, plus ``vertex_proposals`` when enabled.
Downstream label adaptation happens only after final semantic assignment. Its
own loss uses original driver truth. Set ``adapt_labels: false`` on the stage
when downstream truth adaptation is unnecessary.
