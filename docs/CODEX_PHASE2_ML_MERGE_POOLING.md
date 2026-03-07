# Phase2: 学習ベースのマージ（Graph/Reaction Network Pooling）

Phase2は “マージを学習” して、縮退レベルを自動で出すフェーズです。  
ここでの主役は **割当行列 S（species→cluster）** を学習するPoolingです。

---

## 0. 目標成果物
- 反応ネットワーク（ν, F̄, incidence）から学習して S を出す
- S を変えると縮退レベル（弱/中/強）が変わる（Pareto）
- ルールベース制約（状態種/元素族/表面区別など）を “禁止/罰則” で組み込める
- 学習済み S を使い、Phase1の妥当性評価ハーネスで一括検証できる

---

## 1. 推奨ライブラリ
- torch / torch_geometric
- tgp (torch-geometric-pool)  
  BN-Pool（クラスタ数を自動で決める）を使える: BN-PoolはDirichlet Processでクラスタ数を適応決定する。  
  ※Dense poolingなので、巨大機構は Phase3 の剪定後に適用するのが現実的。

---

## 2. グラフ表現（ハイパー反応を扱う方法）
反応は多対多なので **ハイパーグラフ**が自然だが、実装容易性と既存ライブラリ適合のためまずは **二部グラフ**を推奨。

### 2.1 二部（species- reaction）グラフ
- node type:
  - species nodes: Ns
  - reaction nodes: Nr
- edges:
  - species i -> reaction j : if ν[i,j] != 0（符号/係数をedge featureに）
  - reaction j -> species i : 同上（方向を分けると便利）

このhetero graphでmessage passing → species embedding を作る。

### 2.2 species同士のグラフ（簡易）
- adjacency = f(F̄)（種間フラックスの上位k%だけエッジ化）
- こちらはtgpが素直に使えるが、反応ノード情報が落ちる

実装は両方用意し、ベンチで比較できるようにする。

---

## 3. 入力特徴（多条件 + 位相別を強く推奨）
species node features:
- element counts (A[:,i])、元素族one-hot
- phase(one-hot): gas/surface
- time-series統計:
  - mean/max of x_i(t)
  - mean/max of |wdot_i(t)|
  - phase別ピーク/積分
- 反応経路ログ由来の特徴（任意）:
  - 出現頻度、同時出現、直後に出る反応の分布 など

reaction node features（使う場合）:
- ∫|rop_j(t)|dt、phase別
- Arrheniusパラメータ（取れるなら）
- “壁反応/表面反応/気相”のタグ

---

## 4. PoolingでSを学習する
### 4.1 最短ルート（tgp BNPool）
- species graph を作り、BNPoolで S を得る
- loss:
  - downstream reconstruction loss（グラフ再構成 or F̄再構成）
  - + cluster数ペナルティ（BNPoolが内包）
  - + 禁止/罰則制約

### 4.2 禁止/罰則制約（マージ定義の実装）
- hard禁止（mask）:
  - gas vs surface は同クラスタ禁止
  - ion vs neutral は同クラスタ禁止（あるいは罰則）
- soft罰則:
  - neutral↔radical跨ぎは罰則
  - 元素族が違うのは強い罰則
  - 役割タグが違うのは罰則

実装は「割当確率 S_hat」を使い、
- `penalty = Σ_{i,k} Σ_{i',k} S_hat[i,k]*S_hat[i',k]*cost(i,i')`
のような形でペアワイズ罰則を入れる（cost行列は事前計算）。

---

## 5. Sから縮退機構（代表ノード）へ
- y = S^T x  
- dy/dt = S^T wdot（教師データとして使える）

Phase2ではまず
- “縮退後の状態変数 y(t) が情報を保つか”
を重視し、代表反応の再合成は Phase3/Phase4 で強化する。

---

## 6. 学習と評価の流れ
1) Phase0で dataset + network を作成  
2) Phase2で poolingを学習（Sを出す）  
3) Phase1のvalidateで  
   - y(t)の誤差（元のxをpoolして得た y と、縮退系シミュレーションの y）
   - KPI誤差
   - 保存則  
   を自動評価
4) cluster数と誤差のParetoを出す

---

## 7. 実装タスク（コード）
- `cantera_model/reduction/pooling/`
  - `graphs.py` : species graph / bipartite graph builder
  - `features.py`: feature extractor
  - `constraints.py`: mask & penalty
  - `models.py`: GNN encoder + BNPool/SpaPool wrappers
  - `train.py`: training loop（LightningでもOK）
  - `export.py`: Sの保存/読み込み
- `configs/pooling_bn.yaml`
- `tests/test_pooling_constraints.py`

