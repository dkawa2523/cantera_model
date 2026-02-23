import numpy as np

from cantera_model.reduction.pooling.graphs import build_bipartite_graph, build_species_graph


def _species_meta() -> list[dict]:
    return [
        {"name": "CH4", "composition": {"C": 1, "H": 4}},
        {"name": "CH", "composition": {"C": 1, "H": 1}},
        {"name": "CF4", "composition": {"C": 1, "F": 4}},
        {"name": "F", "composition": {"F": 1}},
        {"name": "N", "composition": {"N": 1}},
        {"name": "H", "composition": {"H": 1}},
    ]


def test_pooling_graph_builders_shape_contract() -> None:
    meta = _species_meta()
    ns = len(meta)
    nr = 8
    nu = np.zeros((ns, nr), dtype=float)
    for j in range(nr):
        nu[j % ns, j] = -1.0
        nu[(j + 1) % ns, j] = 1.0

    f_bar = np.zeros((ns, ns), dtype=float)
    f_bar[0, 1] = 1.0
    f_bar[2, 3] = 0.8

    g_species = build_species_graph(nu, f_bar, meta, {})
    assert g_species["type"] == "species_graph"
    assert np.asarray(g_species["adjacency"]).shape == (ns, ns)
    assert np.asarray(g_species["edge_index"]).shape[0] == 2

    rop_stats = np.linspace(0.1, 1.0, nr)
    g_bi = build_bipartite_graph(nu, rop_stats, meta, {})
    assert g_bi["type"] == "bipartite_graph"
    assert int(g_bi["num_species"]) == ns
    assert int(g_bi["num_reactions"]) == nr
    assert np.asarray(g_bi["species_index"]).shape == np.asarray(g_bi["reaction_index"]).shape
