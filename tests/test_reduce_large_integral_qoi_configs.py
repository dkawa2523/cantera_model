from pathlib import Path

import yaml


_CONFIGS = [
    ("configs/reduce_diamond_benchmarks_large_baseline.yaml", ["H2", "CH4", "t_c6HM"], ["C(d)"]),
    ("configs/reduce_diamond_benchmarks_large_learnckpp.yaml", ["H2", "CH4", "t_c6HM"], ["C(d)"]),
    ("configs/reduce_diamond_benchmarks_large_pooling.yaml", ["H2", "CH4", "t_c6HM"], ["C(d)"]),
    ("configs/reduce_sif4_benchmark_sin3n4_large_baseline.yaml", ["HF", "F", "t_HN_SIF(S)"], ["SI(D)", "N(D)"]),
    ("configs/reduce_sif4_benchmark_sin3n4_large_learnckpp.yaml", ["HF", "F", "t_HN_SIF(S)"], ["SI(D)", "N(D)"]),
    ("configs/reduce_sif4_benchmark_sin3n4_large_pooling.yaml", ["HF", "F", "t_HN_SIF(S)"], ["SI(D)", "N(D)"]),
    ("configs/reduce_ac_benchmark_large_baseline.yaml", ["H2", "C2H2", "H_T(s)"], ["C_bulk", "C_D(s)"]),
    ("configs/reduce_ac_benchmark_large_learnckpp.yaml", ["H2", "C2H2", "H_T(s)"], ["C_bulk", "C_D(s)"]),
    ("configs/reduce_ac_benchmark_large_pooling.yaml", ["H2", "C2H2", "H_T(s)"], ["C_bulk", "C_D(s)"]),
]


def test_reduce_large_configs_include_adaptive_kfold_and_integral_qoi() -> None:
    for path_str, species_int_expected, dep_int_expected in _CONFIGS:
        cfg = yaml.safe_load(Path(path_str).read_text())
        qoi = dict(cfg.get("qoi") or {})
        eval_cfg = dict(cfg.get("evaluation") or {})
        split_cfg = dict(eval_cfg.get("surrogate_split") or {})
        policy = dict(split_cfg.get("kfold_policy") or {})

        assert split_cfg.get("mode") == "adaptive_kfold", path_str
        assert int(policy.get("min_cases_for_kfold", -1)) == 4, path_str
        assert int(policy.get("default_k", -1)) == 2, path_str
        assert dict(policy.get("k_by_case_count") or {}) == {6: 3, 8: 3}, path_str

        assert list(qoi.get("species_integral") or []) == species_int_expected, path_str
        assert list(qoi.get("deposition_integral") or []) == dep_int_expected, path_str
