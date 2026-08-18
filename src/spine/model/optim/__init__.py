"""Model optimizer configuration factories.

Optimizer implementations are intentionally not imported here. Importing this
package is part of loading :class:`spine.driver.Driver`, which must remain
available in installations that do not include the optional PyTorch runtime.
"""

from .factory import lr_sched_factory, optim_factory

__all__ = ["lr_sched_factory", "optim_factory"]
