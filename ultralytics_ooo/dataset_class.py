"""The installed dataset subclass, in its own top-level module.

Kept out of ``installer.py`` so that ``import ultralytics_ooo`` (and ``import ultralytics_ooo.core``)
does NOT drag stock Ultralytics into ``sys.modules`` -- the core zero-dependency check relies on that.
The module is only imported when ``install()`` runs, and by spawn DataLoader workers that unpickle the
already-constructed dataset (they re-import this module, whose top-level imports then resolve because
the worker inherits the parent's ``sys.path``).
"""

from __future__ import annotations

from ultralytics.data.dataset import YOLODataset

from ultralytics_ooo.pool.dataset import OnlinePoolDataset


class InstalledYOLODataset(OnlinePoolDataset, YOLODataset):
    """YOLODataset + the mixed virtual-sample pool, with no upstream edits."""
