from pathlib import Path

from cantera_model.cli.reduce_validate import _load_metric_taxonomy_profile


def test_load_metric_taxonomy_profile_from_shared_yaml(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "metric_taxonomy_profiles.yaml"
    taxonomy_path.write_text(
        "\n".join(
            [
                "profiles:",
                "  large_default_v1:",
                "    family_exact:",
                "      T_max: T",
                "    family_prefix:",
                "      \"X_last:\": X_last",
                "    species_token:",
                "      delimiter: ':'",
                "      take: after_first",
                "    metric_family_abs_floor:",
                "      X_last: 1.0e-9",
                "      default: 1.0e-12",
                "",
            ]
        )
    )
    eval_cfg = {
        "metric_taxonomy": {
            "source": "shared_yaml",
            "path": str(taxonomy_path),
            "profile": "large_default_v1",
        }
    }
    resolved = _load_metric_taxonomy_profile(
        eval_cfg,
        config_parent=tmp_path,
        contract={"enforce": True},
    )

    assert resolved["source"] == "shared_yaml"
    assert resolved["profile"] == "large_default_v1"
    assert resolved["family_exact"]["T_max"] == "T"
    assert resolved["species_token"]["delimiter"] == ":"
    assert "metric_taxonomy_resolved" in eval_cfg


def test_load_metric_taxonomy_profile_falls_back_to_legacy_without_enforce(tmp_path: Path) -> None:
    eval_cfg = {
        "metric_taxonomy": {
            "source": "shared_yaml",
            "path": str(tmp_path / "missing.yaml"),
            "profile": "missing",
        }
    }
    resolved = _load_metric_taxonomy_profile(
        eval_cfg,
        config_parent=tmp_path,
        contract={"enforce": False},
    )

    assert resolved["source"] == "legacy_builtin"
    assert resolved["profile"] == "legacy_builtin"
