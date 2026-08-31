UResNet Point-Proposal Model
============================

``model.name: uresnet_ppn`` combines a UResNet segmentation backbone with a
particle-point proposal head, a vertex proposal head, or both.  This page is
written explicitly because the point-proposal package uses runtime PyTorch
types which cannot be imported in Read the Docs' mocked-dependency process.

Network
-------

.. py:class:: spine.model.uresnet.ppn.UResNetPPN(uresnet, ppn=None, vertex=None, proposal_decoder=None)

   Combine UResNet with one or both point-proposal tasks.

   :param dict uresnet: UResNet backbone configuration.
   :param dict ppn: Optional particle-point proposal configuration.
   :param dict vertex: Optional interaction-vertex proposal configuration.
   :param dict proposal_decoder: Optional cross-task decoder configuration.
      Set ``shared: true`` to share the decoder when both proposal tasks are
      configured.

   At least one of ``ppn`` and ``vertex`` is required.  Decoder sharing is
   valid only when both tasks are present.

Loss
----

.. py:class:: spine.model.uresnet.ppn.UResNetPPNLoss(uresnet, uresnet_loss, ppn=None, ppn_loss=None, vertex=None, vertex_loss=None, proposal_decoder=None)

   Supervise segmentation and each configured proposal task.

   :param dict uresnet: UResNet backbone configuration shared with the model.
   :param dict uresnet_loss: Segmentation-loss configuration.
   :param dict ppn: Optional particle-point model configuration.
   :param dict ppn_loss: Required loss configuration when ``ppn`` is enabled.
   :param dict vertex: Optional vertex model configuration.
   :param dict vertex_loss: Required loss configuration when ``vertex`` is enabled.
   :param dict proposal_decoder: Decoder-sharing configuration forwarded by
      the model manager.
