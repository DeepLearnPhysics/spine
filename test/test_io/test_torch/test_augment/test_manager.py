"""Tests for the augmentation manager."""

from spine.io.torch.augment.manager import AugmentManager

from .helpers import BOX2, CropAugment, MaskAugment, make_meta, make_tensor, np, pytest


def test_manager_requires_at_least_one_module():
    with pytest.raises(ValueError):
        AugmentManager()


def test_manager_rejects_non_dict_config():
    with pytest.raises(ValueError):
        AugmentManager(crop="bad")


def test_manager_rejects_unknown_module_name():
    with pytest.raises(ValueError):
        AugmentManager(crop={"name": "unknown"})


def test_manager_skips_none_configs_but_requires_enabled_module():
    with pytest.raises(ValueError):
        AugmentManager(crop=None)


def test_manager_supports_custom_label_with_name():
    manager = AugmentManager(
        first_crop={
            "name": "crop",
            "min_dimensions": BOX2,
            "max_dimensions": BOX2,
            "keep_meta": True,
        }
    )
    assert len(manager.modules) == 1
    assert isinstance(manager.modules[0], CropAugment)


def test_manager_returns_input_if_no_augmented_products():
    manager = AugmentManager(mask={"min_dimensions": BOX2, "max_dimensions": BOX2})
    data = {"foo": 123}
    assert manager(data) is data


def test_manager_rejects_mismatched_metadata():
    manager = AugmentManager(mask={"min_dimensions": BOX2, "max_dimensions": BOX2})
    meta1 = make_meta()
    meta2 = make_meta(lower=(1.0, 0.0, 0.0), upper=(3.0, 4.0, 2.0))
    data = {"a": make_tensor([[0, 0, 0]], meta1), "b": make_tensor([[0, 0, 0]], meta2)}

    with pytest.raises(ValueError):
        manager(data)


def test_manager_copy_meta_detaches_arrays():
    meta = make_meta()
    copy = AugmentManager.copy_meta(meta)
    assert copy is not meta
    assert np.array_equal(copy.lower, meta.lower)
    copy.lower[0] = -1.0
    assert meta.lower[0] != -1.0


def test_manager_applies_modules_in_order_and_updates_context_meta():
    manager = AugmentManager(
        crop={
            "min_dimensions": BOX2,
            "max_dimensions": BOX2,
            "lower": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "upper": np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
            "keep_meta": False,
        },
        mask={
            "min_dimensions": BOX2,
            "max_dimensions": BOX2,
            "lower": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "upper": np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        },
    )
    meta = make_meta(lower=(0.0, 0.0, 0.0), upper=(4.0, 4.0, 4.0))
    data = {
        "voxels": make_tensor([[0, 0, 0], [1, 1, 1], [3, 3, 3]], meta),
        "meta": meta,
    }

    result = manager(data)
    assert result["meta"].count.shape == (3,)
    assert len(result["voxels"].coords) <= 2
