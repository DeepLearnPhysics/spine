from __future__ import annotations

import pytest

from spine.post.optical.flash_matching import FlashMatchProcessor


def test_flash_match_processor_validates_volume():
    with pytest.raises(ValueError, match="volume"):
        FlashMatchProcessor(flash_key="flashes", volume="bad")
