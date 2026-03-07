# Codex Prompts（このまま貼って実装を進める）

> 使い方：  
> 1) Codexにこのリポジトリを開かせる  
> 2) まず `CODEX_CONTEXT.md` と、このファイル（PROMPTS）と、各Phaseのmd を読ませる  
> 3) 下の Prompt を順番に実行

---

## Prompt 0: リポジトリ現状把握（必須）
あなたはこのリポジトリに新機能を実装するソフトウェアエンジニアです。
まず以下を行ってください。

1. リポジトリのディレクトリ構造を一覧化
2. Cantera自動実行コードの入口（CLI/スクリプト/関数）を特定
3. ベンチマークケース（入力機構・条件・期待出力）の場所を特定
4. 既存の出力形式（CSV/npz/jsonなど）を確認し、データスキーマ案を提案
5. 既存のテスト/CIがあれば確認

出力は `docs/repo_audit.md` を新規作成し、上記を箇条書きでまとめてください。
コードは変更しなくてよいです。

---

## Prompt 1: Phase0（データ契約とネットワーク生成）の実装
以下のmdを仕様として実装してください：
- `docs/CODEX_CONTEXT.md`
- `docs/CODEX_PHASE0_DATA_NETWORK.md`

実装要件：
- `CaseResult`（1ケースの結果）と `CaseBundle`（複数ケース）を実装
- 既存のCantera出力を読み込んで `CaseResult` を生成できるI/Oを実装（既存形式に合わせる）
- `nu`（化学量論行列）、`A`（元素行列）を生成する `StoichBuilder` を実装
- `FluxBuilder` を実装し、F̄（位相別でも可）と反応重要度を計算できるようにする
- `pytest` のユニットテストを追加
- 依存追加が必要なら `pyproject.toml` または `requirements.txt` を更新

完了条件：
- ベンチマーク1ケースで `build_network` が動き、nu/A/F̄/I_reaction を保存できる
- テストが通る

---

## Prompt 2: Phase1（ベースライン縮退 + 妥当性評価）の実装
以下のmdを仕様として実装してください：
- `docs/CODEX_PHASE1_BASELINE_REDUCTION_EVAL.md`

実装要件：
- ルールベースmerge（pool_key→P）を実装（設定はYAML）
- 閾値pruneを実装（I_reactionを利用）
- validateハーネスを実装：
  - 時系列誤差（x, wdot）
  - 保存則（元素・表面サイト）
  - KPI（ALD/CVD用にプラガブルに）
- `reports/run_id/` にレポートを自動生成

完了条件：
- ベンチマークで「縮退→再計算→評価→レポート」が一括で回る
- ルールmergeとpruneの縮退率（種数/反応数）がログに出る

---

## Prompt 3: Phase2（PoolingでSを学習）の最小実装（BNPool）
以下のmdを仕様として実装してください：
- `docs/CODEX_PHASE2_ML_MERGE_POOLING.md`

実装要件：
- species graph builder を実装（F̄の上位エッジでグラフ化）
- node feature extractor を実装（位相別統計を含む）
- tgp の BNPool を用いて S を学習する `train_pooling_bn.py` を実装
- 制約（mask/penalty）を実装し、設定でON/OFF可能に
- 学習済みSを保存し、Phase1 validate に渡せるようにする

完了条件：
- ベンチマークで学習が回り、クラスタ数が自動で決まる
- Sを使って代表状態 y を生成し、validateで誤差が見られる

---

## Prompt 4: Phase3-4（強い縮退 / 最大縮退）の実装（優先順）
以下のmdを仕様として、**優先順に** 実装してください：
- `docs/CODEX_PHASE3_4_ADVANCED.md`

優先順：
1) prune学習（hard-concrete gate）で反応数を削る（Phase4A）
2) LearnCK++（候補overall反応 + スパース選択 + ANN-hard射影）（Phase3A）
3) CRNN/SRNN（表面対応）（Phase3B）
4) SINDy/CRN再構成（Phase4B）

完了条件：
- 1) がベンチマークで反応数削減と誤差トレードオフを示せる
- 2) が “少数代表反応” で y(t) を良好に再現できる
- すべて Phase1 validate に統合され、比較レポートが出る

