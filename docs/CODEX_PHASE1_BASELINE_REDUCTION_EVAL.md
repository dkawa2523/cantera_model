# Phase1: ベースライン縮退（ルールベースmerge + 閾値prune）と妥当性評価

Phase1は「まず動く縮退パイプライン」を作るフェーズです。  
MLはPhase2以降でも、**ベースラインがないと効果が測れない**ため最優先です。

---

## 0. 目標成果物
- ルールベースで **代表ノード（任意状態）** を作り、種数を減らす
- 重要度閾値で反応をpruneし、反応数を減らす
- 縮退後の機構を使ってCantera再計算し、**誤差と保存則**を自動評価
- ここまでをCLIで回せる

---

## 1. ルールベースmerge（最小実装）
### 1.1 代表ノード（pool）のキー設計（例）
species i に対して `pool_key(i)` を作る：

- phase: gas / surface / bulk（Cantera phase名で判別）
- charge_state: ion / neutral（電荷が取れるなら）
- radical_flag: radical / nonradical（取れなければ“未定”でよい）
- element_family: 元素集合（{C,H},{C,F},{Si,Cl,O}…）
- role_tag: user-defined（成膜/エッチ/キャリア etc; YAMLでマッピング）

pool_key = (phase, charge_state, radical_flag, element_family, role_tag)

> “CH4とCHをマージしたい” を叶えるには  
> element_familyが同じで role_tagが同じなら pool同一にできる。  
> ただし neutral/radical跨ぎは config で
> - `allow_cross_radical: true/false`
> - `penalty_cross_radical: float`（Phase2で使う）
> のように設計しておく。

### 1.2 merge写像 P
- `P: (Ns, Npools)` の0/1行列（各speciesがどのpoolに入るか）
- 代表状態 y = P^T x

### 1.3 “代表種”の扱い
Canteraに戻す都合で
- (a) 代表状態はCantera外でODEとして評価（簡単）  
- (b) 代表状態を擬似speciesとして機構YAMLを生成（難しい）  
Phase1では (a) 推奨（まず評価系を作る）。

---

## 2. prune（閾値ベースの最小実装）
- `I_reaction[j]` が小さい反応を削除
- 可逆反応を分解している場合はforward/backwardを一緒に扱うなど、ルールを用意

出力：
- `keep_reactions: bool[Nr]`
- `keep_species: bool[Ns]`（必要なら）

---

## 3. 妥当性評価（validation harness）
### 3.1 比較する量
- x(t)（濃度/モル分率）: RMSE / max / phase別
- wdot(t): RMSE / max / sign一致率
- rop(t): 重要反応だけ比較（全部比較すると重い）

### 3.2 KPI（半導体プロセス向けに“外だし”で定義）
`kpi.py` に “プロセス別KPI計算器” を追加できるようにする。

例（ALD）：
- GPC（成膜量/サイクル）
- 飽和到達時間
- 副生成物積算量
- ラジカルピーク（phase別最大値）
- 表面被覆率（site balance）

### 3.3 保存則チェック
- 元素: `A @ x(t)` の差分
- 電荷（あるなら）
- 表面サイト: Σcoverages=1 など

---

## 4. レポート出力
- `reports/{run_id}/summary.json`
- `reports/{run_id}/metrics.csv`
- `reports/{run_id}/plots/*.png`
- `reports/{run_id}/reduction_map.json`（pool_key, P, keep_reactions）

---

## 5. CLI（例）
- `python -m cantera_model.reduce baseline --config configs/reduce_baseline.yaml`
- `python -m cantera_model.validate --config configs/validate.yaml --run_id ...`

---

## 6. テスト
- merge後の P が各行1つだけ1（hard assignment）
- merge後でも element balance 指標が定義される
- pruneで反応が減ってもデータ構造が壊れない

