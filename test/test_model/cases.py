"""Supported model configurations used by model contract tests."""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
EXAMPLE_CONFIGS = sorted(CONFIG_DIR.rglob("*.cfg")) + sorted(CONFIG_DIR.rglob("*.yaml"))

MODEL_CONFIGS = {
    "full_chain": CONFIG_DIR / "full_chain" / "full_chain_train.yaml",
    "graph_spice": CONFIG_DIR / "graph_spice" / "graph_spice_train.yaml",
    "grappa_inter": CONFIG_DIR / "grappa_inter" / "grappa_inter_train.yaml",
    "grappa_shower": CONFIG_DIR / "grappa_shower" / "grappa_shower_train.yaml",
    "grappa_track": CONFIG_DIR / "grappa_track" / "grappa_track_train.yaml",
    "image_energy": CONFIG_DIR / "image" / "energy" / "image_energy_train.yaml",
    "image_energy_ancestor": (
        CONFIG_DIR / "image" / "energy" / "image_energy_ancestor_train.yaml"
    ),
    "image_pid": CONFIG_DIR / "image" / "pid" / "image_pid_train.yaml",
    "image_pid_ancestor": (
        CONFIG_DIR / "image" / "pid" / "image_pid_ancestor_train.yaml"
    ),
    "spice": CONFIG_DIR / "spice" / "spice_train.yaml",
    "uresnet": CONFIG_DIR / "uresnet" / "uresnet_train.yaml",
    "uresnet_bayes": (CONFIG_DIR / "uresnet" / "bayes" / "uresnet_bayes_train.yaml"),
    "uresnet_ppn": CONFIG_DIR / "uresnet" / "ppn" / "uresnet_ppn_train.yaml",
}

INFERENCE_MODEL_CONFIGS = {
    "full_chain": CONFIG_DIR / "full_chain" / "full_chain_test.yaml",
    "graph_spice": CONFIG_DIR / "graph_spice" / "graph_spice_test.yaml",
    "grappa_inter": CONFIG_DIR / "grappa_inter" / "grappa_inter_test.yaml",
    "grappa_shower": CONFIG_DIR / "grappa_shower" / "grappa_shower_test.yaml",
    "grappa_track": CONFIG_DIR / "grappa_track" / "grappa_track_test.yaml",
    "image_energy": CONFIG_DIR / "image" / "energy" / "image_energy_test.yaml",
    "image_energy_ancestor": (
        CONFIG_DIR / "image" / "energy" / "image_energy_ancestor_test.yaml"
    ),
    "image_pid": CONFIG_DIR / "image" / "pid" / "image_pid_test.yaml",
    "image_pid_ancestor": (
        CONFIG_DIR / "image" / "pid" / "image_pid_ancestor_test.yaml"
    ),
    "spice": CONFIG_DIR / "spice" / "spice_test.yaml",
    "uresnet": CONFIG_DIR / "uresnet" / "uresnet_test.yaml",
    "uresnet_bayes": (CONFIG_DIR / "uresnet" / "bayes" / "uresnet_bayes_test.yaml"),
    "uresnet_ppn": CONFIG_DIR / "uresnet" / "ppn" / "uresnet_ppn_test.yaml",
}

FULL_CHAIN_REGRESSION_CONFIG = CONFIG_DIR / "full_chain" / "full_chain_regression.yaml"

POINT_PROPOSAL_VARIANT_CONFIGS = {
    "vertex": (
        CONFIG_DIR / "uresnet" / "ppn" / "uresnet_vertex_train.yaml",
        CONFIG_DIR / "uresnet" / "ppn" / "uresnet_vertex_test.yaml",
    ),
    "ppn_vertex": (
        CONFIG_DIR / "uresnet" / "ppn" / "uresnet_ppn_vertex_train.yaml",
        CONFIG_DIR / "uresnet" / "ppn" / "uresnet_ppn_vertex_test.yaml",
    ),
}

STANDALONE_MODEL_CONFIGS = {
    name: path for name, path in MODEL_CONFIGS.items() if name != "full_chain"
}

EXPECTED_OUTPUTS = {
    "graph_spice": {
        "coordinates",
        "features",
        "filter_index",
        "edge_index",
        "edge_attr",
        "edge_prob",
    },
    "grappa_inter": {"clusts", "edge_index"},
    "grappa_shower": {"clusts", "edge_index"},
    "grappa_track": {"clusts", "edge_index"},
    "image_energy": {"objects", "energy_pred"},
    "image_energy_ancestor": {"objects", "energy_pred"},
    "image_pid": {"objects", "pid_pred"},
    "image_pid_ancestor": {"objects", "pid_pred"},
    "spice": {"embeddings", "margins", "seediness", "filter_index"},
    "uresnet": {"segmentation"},
    "uresnet_bayes": {"segmentation"},
    "uresnet_ppn": {"segmentation", "ppn_points"},
}
