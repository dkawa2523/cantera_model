# standalone/cantera_gri30_eval

GRI-Mech 3.0 の Cantera 実行・評価だけを独立で回すための最小パックです。
`rxn_platform` の task/pipeline 実装には依存しません。

## 含まれるもの
- `run_cantera_eval.py`:
  - 条件CSVを自動実行
  - baseline/candidate の QoI 比較
  - `summary.json`, `metrics.json`, `comparison_results.csv` を出力
- `cantera_model/`:
  - `eval/cantera_runner.py`: 既存 evaluator のモジュール化本体
  - `cli/run_cantera_trace.py`: Cantera時系列トレースをHDF5へ保存
  - `cli/build_network.py`: trace_h5 から `nu/A/F_bar/I_reaction` を生成
  - `reduction/merge_free.py`: 共通元素制約付きの自由マージ
  - `reduction/prune_gate.py`: hard-concrete 風ゲート pruning
  - `reduction/conservation.py`: 保存則射影
  - `eval/surrogate_eval.py`: surrogate 候補評価
  - `cli/reduce_validate.py`: Stage A/B/C の縮退探索 + gate 判定 + レポート出力
- `assets/mechanisms/gri30.yaml`: GRI-Mech 3.0
- `assets/conditions/`:
  - `gri30_small.csv`
  - `gri30_netbench_val.csv`
  - `gri30_tiny.csv`（動作確認用）
- `configs/`:
  - `gri30_small_baseline.yaml`
  - `gri30_small_compare_template.yaml`
  - `gri30_tiny_quick.yaml`
- `setup_env.sh`: venv + pip で実行環境を自動構築
- `environment.yml`: conda/mamba 用の環境定義
- `requirements.txt` / `requirements-lock.txt`: pip 依存定義

## 依存
- Python 3.10+
- `cantera`
- `pyyaml`

## 環境構築

### venv + pip（推奨）
```bash
cd standalone/cantera_gri30_eval
./setup_env.sh
```

任意で Python バイナリと venv 先を指定:
```bash
PYTHON_BIN=python3.11 VENV_DIR=.venv ./setup_env.sh
```

### conda / mamba
```bash
cd standalone/cantera_gri30_eval
mamba env create -f environment.yml
conda activate cantera-gri30-eval
```

## 実行
```bash
cd standalone/cantera_gri30_eval
python3 run_cantera_eval.py --config configs/gri30_small_baseline.yaml
python3 run_cantera_eval.py --config configs/gri30_small_compare_template.yaml
python3 run_cantera_eval.py --config configs/diamond_benchmarks_diamond_quick.yaml --run-id diamond_benchmarks_diamond_demo
python3 run_cantera_eval.py --config configs/diamond_benchmarks_diamond_large_quick.yaml --run-id diamond_benchmarks_diamond_large_demo
python3 run_cantera_eval.py --config configs/sif4_benchmark_sin3n4_large_quick.yaml --run-id sif4_benchmark_large_demo
python3 run_cantera_eval.py --config configs/ac_benchmark_large_quick.yaml --run-id ac_benchmark_large_demo
./run_baseline.sh
./run_compare.sh
PYTHON_BIN=/path/to/python ./run_compare.sh --run-id my_run
```

### Aggressive Reduction（surrogate-first）
```bash
python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_surrogate_aggressive.yaml \
  --run-id aggressive_demo
```

### Cantera Trace 生成（実データ入力）
```bash
python3 run_cantera_trace.py \
  --config configs/gri30_tiny_trace.yaml \
  --run-id tiny_trace
```

`reduce_validate` の設定で `trace_h5` を指定すると、合成データではなく実トレースを使って縮退探索を実行します。

### Surface Trace（benchmarks_diamond）
```bash
python3 -m cantera_model.cli.run_surface_trace \
  --config configs/diamond_benchmarks_diamond_trace.yaml \
  --run-id diamond_benchmarks_diamond_trace

python3 -m cantera_model.cli.run_surface_trace \
  --config configs/diamond_benchmarks_diamond_large_trace.yaml \
  --run-id diamond_benchmarks_diamond_large_trace

python3 -m cantera_model.cli.run_surface_trace \
  --config configs/sif4_benchmark_sin3n4_large_trace.yaml \
  --run-id sif4_benchmark_large_trace

python3 -m cantera_model.cli.run_surface_trace \
  --config configs/ac_benchmark_large_trace.yaml \
  --run-id ac_benchmark_large_trace
```

`diamond_benchmarks_diamond_large_trace.yaml` は `include_gas_reactions_in_trace: true` を有効化し、
trace の反応次元を gas(325)+surface(60)=385 に拡張します。
このとき `wdot` は `trace_wdot_policy: stoich_consistent` により `wdot = nu @ rop` で保存されます。

### benchmarks_diamond の縮退評価（3モード比較）
```bash
python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_diamond_benchmarks_baseline.yaml \
  --run-id reduce_diamond_benchmarks_baseline

python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_diamond_benchmarks_learnckpp.yaml \
  --run-id reduce_diamond_benchmarks_learnckpp

python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_diamond_benchmarks_pooling.yaml \
  --run-id reduce_diamond_benchmarks_pooling

python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_diamond_benchmarks_large_baseline.yaml \
  --run-id reduce_diamond_benchmarks_large_baseline

python3 -m cantera_model.cli.summarize_reduction_eval \
  --entry baseline:reduce_diamond_benchmarks_baseline \
  --entry learnckpp:reduce_diamond_benchmarks_learnckpp \
  --entry pooling:reduce_diamond_benchmarks_pooling \
  --output reports/diamond_benchmarks_diamond_eval_summary.json
```

`diamond_benchmarks_diamond_large_quick.yaml` の `summary.json` では
`baseline_counts.species=77`, `baseline_counts.reactions=385` を確認できます（gas+surface合算）。

`sif4_benchmark_sin3n4_large_quick.yaml` の `summary.json` では
`baseline_counts.species>=50`, `baseline_counts.reactions>=100` を確認できます（gas+surface合算）。

`ac_benchmark_large_quick.yaml` の `summary.json` では
`baseline_counts.species>=50`, `baseline_counts.reactions>=100` を確認できます（gas+surface合算）。

### benchmark_sif4_sin3n4_cvd の縮退評価（3モード比較）
```bash
python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_sif4_benchmark_sin3n4_large_baseline.yaml \
  --run-id reduce_sif4_large_baseline

python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_sif4_benchmark_sin3n4_large_learnckpp.yaml \
  --run-id reduce_sif4_large_learnckpp

python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_sif4_benchmark_sin3n4_large_pooling.yaml \
  --run-id reduce_sif4_large_pooling

python3 -m cantera_model.cli.summarize_reduction_eval \
  --entry baseline:reduce_sif4_large_baseline \
  --entry learnckpp:reduce_sif4_large_learnckpp \
  --entry pooling:reduce_sif4_large_pooling \
  --output reports/sif4_benchmark_sin3n4_large_eval_summary.json
```

### benchmark_large (ac_hydrocarbon_cvd_large) の縮退評価（3モード比較）
```bash
python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_ac_benchmark_large_baseline.yaml \
  --run-id reduce_ac_large_baseline

python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_ac_benchmark_large_learnckpp.yaml \
  --run-id reduce_ac_large_learnckpp

python3 -m cantera_model.cli.reduce_validate \
  --config configs/reduce_ac_benchmark_large_pooling.yaml \
  --run-id reduce_ac_large_pooling

python3 -m cantera_model.cli.summarize_reduction_eval \
  --entry baseline:reduce_ac_large_baseline \
  --entry learnckpp:reduce_ac_large_learnckpp \
  --entry pooling:reduce_ac_large_pooling \
  --output reports/ac_benchmark_large_eval_summary.json
```

### ネットワークアーティファクト生成
```bash
python3 run_build_network.py \
  --trace-h5 artifacts/traces/tiny_trace.h5 \
  --run-id tiny_network
```

phase分割を使う場合:
```bash
python3 run_build_network.py \
  --trace-h5 artifacts/traces/tiny_trace.h5 \
  --run-id tiny_network_phase \
  --phase-fractions "pulse:0.4,purge:0.6"
```

状態時系列アーティファクトを保存しない場合:
```bash
python3 run_build_network.py \
  --trace-h5 artifacts/traces/tiny_trace.h5 \
  --run-id tiny_network_no_state \
  --save-state false
```

出力:
- `artifacts/network/<run_id>/nu.npy`
- `artifacts/network/<run_id>/A.npy`
- `artifacts/network/<run_id>/F_bar.npy`
- `artifacts/network/<run_id>/I_reaction.npy`
- `artifacts/network/<run_id>/F_bar_by_phase.npy`
- `artifacts/network/<run_id>/I_reaction_by_phase.npy`
- `artifacts/network/<run_id>/phase_names.json`
- `artifacts/network/<run_id>/phase_fractions.json`
- `artifacts/network/<run_id>/time.npy`（`--save-state true`）
- `artifacts/network/<run_id>/X.npy`（`--save-state true`）
- `artifacts/network/<run_id>/case_slices.json`（`--save-state true`）

`reduce_validate` の入力優先順位:
1. `network_dir`（最優先）
2. `trace_h5`
3. synthetic fallback

`reduce_validate` の surrogate はデフォルトで `linear_ridge`（条件→QoI軽量回帰）を使用し、
stageごとの差分は `metric_drift` 由来の摂動で与えます。

`network_dir` モードでは `merge.phase_weights` または `merge.phase_select` を設定すると、
`F_bar_by_phase` / `I_reaction_by_phase` を重み付き合成してマージ・pruneへ反映します。
両方を同時に設定した場合は設定不整合としてエラー終了します。

`evaluation` には以下を設定できます。
- `surrogate_split.mode: kfold|in_sample`（既定: `kfold`, ケース不足時は `in_sample` へフォールバック）
- `physical_gate.enabled: true|false`（既定: `true`）
- `physical_gate.max_conservation_violation`
- `physical_gate.max_negative_steps`

## 開発時の検証フロー（推奨）
環境構築:
```bash
./setup_env.sh
```

実装中の回帰確認を一括実行:
```bash
./run_dev_checks.sh
```

このスクリプトは順に以下を実行します。
- `pytest`
- tiny Cantera evaluator smoke
- tiny trace 生成
- tiny network 生成
- `reduce_validate`（synthetic / trace_h5 / network_dir）
- `reduce_validate`（pooling mode）
- `tune_pooling` スモーク（synthetic + trace + network）

主なオプション:
```bash
./run_dev_checks.sh <run_tag> --pooling-trials 2 --pooling-backend pyg
./run_dev_checks.sh --run-tag dev_fast --skip-pooling-tune
```

このモードでは以下を保証します。
- 共通元素ゼロの種ペアは `hard ban`（同クラスタ禁止）
- 共通元素ありは候補化し、`element/fragment/flux/role` スコアで優先度付け
- phase/charge/radical/role 差は `soft penalty` で扱う

## はじめて使う手順（最短）
1. ディレクトリへ移動
```bash
cd standalone/cantera_gri30_eval
```
2. 環境構築（venv）
```bash
./setup_env.sh
```
3. baseline 実行
```bash
./run_baseline.sh --run-id demo_baseline
```
4. candidate 比較実行
```bash
./run_compare.sh \
  --run-id demo_compare \
  --candidate-mechanism ../path/to/candidate.yaml
```
5. 結果確認
```bash
cat runs/demo_compare/summary.json
ls runs/demo_compare
```

### CLI override（YAMLを編集せずに実行）
```bash
python run_cantera_eval.py \
  --config configs/gri30_small_compare_template.yaml \
  --candidate-mechanism ../path/to/candidate.yaml \
  --conditions-csv assets/conditions/gri30_netbench_val.csv \
  --rel-tolerance 0.15 \
  --n-steps 600 \
  --run-id custom_compare_run
```

### candidate 機構ファイルの置き場所例
推奨は、このリポジトリ直下に `models/` を作って置く方法です。

```bash
mkdir -p models
cp /path/to/reduced_gri30.yaml models/reduced_gri30.yaml

./run_compare.sh \
  --run-id compare_reduced \
  --candidate-mechanism models/reduced_gri30.yaml
```

`assets/mechanisms/` に置いて比較しても問題ありません。
```bash
cp /path/to/reduced_gri30.yaml assets/mechanisms/reduced_gri30.yaml
./run_compare.sh --candidate-mechanism assets/mechanisms/reduced_gri30.yaml
```

## 出力
`runs/<run_id>/` に出力されます。
- `summary.json`: 実行サマリ（種数・反応数・比較指標）
- `metrics.json`: `summary.json` と同内容
- `baseline_results.csv`
- `candidate_results.csv`（candidate指定時）
- `comparison_results.csv`（candidate指定時）

## 比較指標
- ignition delay（`dT/dt`最大時刻）
- `T_max`
- 最終組成（`species_last`）
- 最大組成（`species_max`）
- ケース単位 pass（全QoIの相対誤差が `rel_tolerance` 以下）
- `pass_rate`, `max_rel_diff`, `mean_rel_diff`

## summary.json の主要キー
- `status`: `ok` / `failed`
- `failure_reason`: 失敗理由（`missing_file`, `invalid_phase`, `invalid_csv`, `cantera_load_error` など）
- `qoi_metrics_count`: 比較したQoI指標数
- `pass_cases`, `failed_cases`: ケース単位合否
- `worst_case`: 最大相対誤差のケースと指標

## 失敗時の確認
1. `runs/<run_id>/summary.json` の `status` と `failure_reason` を確認
2. `error_message` で入力不正（機構パス、phase名、CSV列不足など）を特定
3. 修正後に同じ `--run-id` か新しい `--run-id` で再実行

## 注意
- この切り出しは「Cantera実行評価」を独立化するためのものです。
- 既存の縮退 task（repair/optimize/viz）は含みません。
