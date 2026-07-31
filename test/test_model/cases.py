"""Supported model configurations used by model contract tests."""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

MODEL_CONFIGS = {
    "full_chain": CONFIG_DIR / "test_full_chain.yaml",
    "graph_spice": CONFIG_DIR / "train_graph_spice.cfg",
    "grappa_inter": CONFIG_DIR / "train_grappa_inter.cfg",
    "grappa_shower": CONFIG_DIR / "train_grappa_shower.yaml",
    "grappa_track": CONFIG_DIR / "train_grappa_track.cfg",
    "image_class": CONFIG_DIR / "train_image_class.cfg",
    "spice": CONFIG_DIR / "spice" / "spice_train.yaml",
    "uresnet": CONFIG_DIR / "uresnet" / "uresnet_train.yaml",
    "uresnet_ppn": CONFIG_DIR / "train_uresnet_ppn.cfg",
}

INFERENCE_MODEL_CONFIGS = {
    "full_chain": CONFIG_DIR / "test_full_chain.yaml",
    "spice": CONFIG_DIR / "spice" / "spice_test.yaml",
    "uresnet": CONFIG_DIR / "uresnet" / "uresnet_test.yaml",
}

STANDALONE_MODEL_CONFIGS = {
    name: path for name, path in MODEL_CONFIGS.items() if name != "full_chain"
}

EXPECTED_OUTPUTS = {
    "graph_spice": {"filter_index"},
    "grappa_inter": {"clusts", "edge_index"},
    "grappa_shower": {"clusts", "edge_index"},
    "grappa_track": {"clusts", "edge_index"},
    "image_class": {"logits"},
    "spice": {"embeddings", "margins", "seediness", "filter_index"},
    "uresnet": {"segmentation"},
    "uresnet_ppn": {"segmentation", "ppn_points"},
}
