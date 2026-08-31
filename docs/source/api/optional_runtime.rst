Optional Model-Runtime APIs
===========================

These APIs require a real PyTorch runtime and therefore cannot be imported by
Read the Docs while optional dependencies are mocked. Their public signatures
and configuration contracts are documented explicitly here.

Full-chain segmentation provider
--------------------------------

The ``segmentation`` full-chain stage accepts ``mode: uresnet`` for learned
segmentation or ``mode: label`` to use semantic truth. In learned mode, provide
exactly one of ``uresnet`` and ``uresnet_ppn``. The optional ``adapt_labels``
mapping configures truth-label alignment; ``point_proposal: ppn`` requires the
combined ``uresnet_ppn`` model.

.. py:class:: spine.model.full_chain.providers.segmentation.SegmentationStage(name, mode, model, label_adapter)

   Produce semantic predictions and optional point proposals for the full
   chain.

   :param str name: Stage name.
   :param str mode: Either ``"uresnet"`` or ``"label"``.
   :param model: Configured UResNet or UResNet-PPN model, or ``None`` in label
      mode.
   :param label_adapter: Adapter used to align structured truth with the
      effective voxel set.

.. py:class:: spine.model.full_chain.providers.segmentation.SegmentationLossStage(name, loss)

   Align truth rows and route supervision to the UResNet or UResNet-PPN loss.

.. py:function:: spine.model.full_chain.providers.segmentation.build_segmentation_stage(name, config, owner)

   Build a segmentation stage from the resolved full-chain ``config`` mapping.

.. py:function:: spine.model.full_chain.providers.segmentation.build_segmentation_loss(name, config, owner)

   Build the corresponding supervised loss stage, or return ``None`` when the
   stage has no ``loss`` block.

AdaBound optimizers
-------------------

Both optimizers accept an iterable of model parameters or optimizer parameter
groups. ``AdaBoundW`` applies weight decay separately from the gradient update.

.. py:class:: spine.model.optim.adabound.AdaBound(params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, eps=1e-8, weight_decay=0, amsbound=False)

.. py:class:: spine.model.optim.adabound.AdaBoundW(params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, eps=1e-8, weight_decay=0, amsbound=False)

   :param params: Parameters to optimize or dictionaries defining parameter
      groups.
   :param float lr: Initial Adam learning rate.
   :param tuple betas: Running-gradient moment coefficients.
   :param float final_lr: Final bounded SGD learning rate.
   :param float gamma: Convergence speed of the dynamic bounds.
   :param float eps: Numerical-stability term.
   :param float weight_decay: Weight-decay coefficient.
   :param bool amsbound: Whether to use the AMSBound variant.
