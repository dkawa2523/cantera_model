# Phase3-4: 強い縮退（LearnCK++/CRNN/SRNN + ANN-hard）と、縮退効果最大化（prune学習 + SINDy/CRN再構成）

---

## Phase3 目標：実用レベルの“強い縮退”を出す
Phase2の S（マージ）だけだと、状態数は減っても「反応式数」が減らない/代表反応が作れないことがある。  
Phase3では **代表反応** を作り、縮退後の系を **独立にシミュレーション可能**にする。

### Phase3A: LearnCK++（overall reactionの学習）
参照：LearnCK（overall reaction + NNで速度を学習、質量保存を組み込む） citeturn11search0turn11search1

#### 実装アイデア（半導体向け拡張）
- “安定種のみ”ではなく、ユーザ要求に合わせて  
  状態種×元素族×役割 で作った **代表状態 y** を使う
- overall reactions は
  - (1) ルールベースで候補生成（F̄の強結合エッジから生成）
  - (2) 学習でスパース選択（L1/L0 gate）
  の2段で「少数反応」を狙う

#### ライブラリ/関連技術（効果最大化）
- L0正則化ゲート: hard-concrete / Louizos式（自前でも可）
- 物理制約: element balance を射影で満たす “ANN-hard” 的補正（予測後に制約面へ射影） citeturn12search0
- 学習安定化:  
  - 反応速度出力は softplus で非負  
  - scale normalization（log1pや標準化）
- 実験設計: optunaで
  - 反応数ペナルティ係数
  - ネットワーク幅
  - 学習率  
  を探索

#### 実装タスク（コード）
- `cantera_model/reduction/learnck/`
  - `candidate_reactions.py` : F̄から候補overall反応生成
  - `model.py` : overall reaction rate predictor（MLP）
  - `constraints.py` : 保存則射影（ANN-hard）
  - `train.py` : dy/dt教師あり学習（NeuralODEなし）
  - `simulate.py` : reduced ODE の forward simulation（scipy）
  - `export.py` : JSON/YAMLで縮退機構を書き出し（まずはCantera外）
- `configs/learnck.yaml`

---

### Phase3B: CRNN / SRNN（反応式を“解釈可能NN”で再構成）
- Atom-conserving CRNN（原子保存層を組み込む） citeturn19search2
- 表面反応には SRNN（表面被覆率補正などの制約を持つ） citeturn12search0

> 半導体装置では表面が重要なので、  
> “気相”はLearnCK++、 “表面”はSRNN という分割も現実的。

#### 実装タスク（コード）
- `cantera_model/reduction/crnn_srnn/`
  - `stoich_param.py` : 反応式（ν）をパラメタ化し原子保存を満たす
  - `rate_law.py` : mass-action + Arrhenius（拡張可能に）
  - `coverage.py` : SRNNのcoverage correction
  - `train.py` : dx/dt or dy/dt の教師あり学習

---

## Phase4 目標：縮退効果（状態数・反応数）を最大化する
Phase4は「元の反応式に縛られない」方向で、  
多条件時系列から最小モデルを再発見する。

### Phase4A: prune学習（削除を“学習で最適化”）
Phase1の閾値pruneを、深層学習ゲートに置き換える。

#### 推奨（実装容易で効く）
- 反応ごとに gate z_j ∈ [0,1] を持ち、
  - wdot_hat = ν @ (z ⊙ rop) を近似
  - 損失 = MSE(wdot_hat, wdot_true) + λ * Σ z_j
- z_j は hard-concrete で “実質L0” を実現
- これを全ケースで学習し、**多条件で本当に効く反応だけ残す**

---

### Phase4B: SINDy（Integral/weak + group sparsity）→ CRN再構成
- 剛性やノイズ対策として、weak/integral系SINDyを優先
- マルチケースで共通構造を取るため group sparsity（GS-SINDyの考え方） citeturn18search1
- 未知パラメータ含む場合、ADAM-SINDyでライブラリ依存を下げる citeturn19search46
- 剛性対策として IRK-SINDy の知見も利用（評価側の数値安定） citeturn18search0

#### ライブラリ/関連技術（効果最大化）
- pysindy（ベースライン）
- cvxpy（group-lasso、CRN再構成の凸最適化）
- 時系列前処理:
  - Savitzky–Golay / total-variation denoising（scipy）
  - 位相別に学習（ALDで特に効く）

#### 実装タスク（コード）
- `cantera_model/reduction/sindy_crn/`
  - `preprocess.py` : 位相分割・正規化・ノイズ低減
  - `library.py` : mass-action対応の項ライブラリ
  - `group_sparse.py` : マルチケース共通項選択（cvxpy）
  - `fit.py` : pysindy or 自前回帰
  - `crn_reconstruct.py` : ODE→反応式への変換（凸最適化）
  - `simulate.py` : 同定モデルのforward simulation
- `configs/sindy.yaml`

---

## Phase3-4 共通：自動評価（Pareto探索）
- 目的関数:
  - minimize: (#species, #reactions)
  - subject to: KPI誤差 < tol, 保存則誤差 < tol
- 実装:
  - optuna / pymoo で探索
  - 1 trial = (削除率, マージ設定, λ) の組
  - trialごとに Phase1 validate を回してスコア化

---

## 出力フォーマット（統一）
- `artifacts/reduction/{run_id}/mapping.json`（S/P、pool_key、keep_reactions）
- `artifacts/reduction/{run_id}/model.pt`（学習モデル）
- `artifacts/reduction/{run_id}/reduced_ode.yaml` or `reduced_crn.yaml`
- `reports/{run_id}/report.md`

