# standalone/cantera_gri30_eval

GRI-Mech 3.0 の Cantera 実行・評価だけを独立で回すための最小パックです。
`rxn_platform` の task/pipeline 実装には依存しません。

## 含まれるもの
- `run_cantera_eval.py`:
  - 条件CSVを自動実行
  - baseline/candidate の QoI 比較
  - `summary.json`, `metrics.json`, `comparison_results.csv` を出力
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
./run_baseline.sh
./run_compare.sh
PYTHON_BIN=/path/to/python ./run_compare.sh --run-id my_run
```

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
