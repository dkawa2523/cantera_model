# Phase0: データ契約（schema）と反応ネットワーク生成

このファイルは **まず最初に実装する最小コア**（全手法共通の土台）です。

---

## 0. 目標成果物
- Canteraの複数ケース出力を **同一形式** で読み込める
- そこから **ν（化学量論行列）** と **フラックス/重要度** を計算できる
- 後段のML/縮退が **このAPIだけに依存** できる

---

## 1. ディレクトリ案（既存構造に合わせて調整）
- `cantera_model/`
  - `data/`
    - `schema.py`  ← dataclass / pydantic
    - `io.py`      ← Cantera結果の読み書き
  - `network/`
    - `stoich.py`  ← ν作成、元素行列A作成
    - `flux.py`    ← species→species flux、reaction importance
    - `phase.py`   ← ALD位相分割（pulse/purge等）
  - `utils/`
    - `sparse.py`  ← sparse共通処理
    - `hashing.py` ← ケースID生成
- `benchmarks/`（既存があれば流用）
- `tests/`

---

## 2. データ契約（推奨：xarray Dataset + メタ情報dict）
### 2.1 1ケース（CaseResult）
最低限入れる（例）：
- `time: (T,)`
- `Tgas: (T,)` / `pressure: (T,)`（一定ならスカラーでもOK）
- `X: (T, Ns)`（species mole fraction or concentration）
- `wdot: (T, Ns)`（species net production rate）
- `rop: (T, Nr)`（reaction rate-of-progress）
- `meta: dict`
  - `case_id`
  - `mechanism_path`
  - `conditions`（温度/圧力/流量/電力/位相情報など）

実装は dataclass でも良いが、後で扱いやすいので
- `CaseResult.to_xarray()` / `CaseResult.from_xarray()` を用意する。

### 2.2 複数ケース（CaseBundle / Dataset）
- ケースを `zarr` で保存し、逐次読み込みできるようにする（巨大化対策）
- 1ケースごとに `case_id` をキーにして保存

---

## 3. 反応ネットワーク生成
### 3.1 化学量論行列 ν
Canteraから機構を読み、`Ns×Nr` の ν を作る。

- reactants は負、products は正
- 可逆反応は「1反応として扱う（ropが正負）」か「forward/backwardで2本に分ける」かを **configで選べる** ようにする  
  ※縮退用途では 1反応扱いが扱いやすいことが多い。

出力:
- `nu: scipy.sparse.csr_matrix` 推奨
- species名リスト、reaction式リスト

### 3.2 元素行列 A（保存則）
- `A: (Ne, Ns)`：元素×species の原子数（Cantera Species.composition から）
- 表面サイト保存も扱うなら `Asite` を別で持つ（site種類×species）

---

## 4. フラックス行列 F（species→species）
目的：**マージ/階層化** の入力に使う “強結合” を定量化する。

### 4.1 反応寄与からのフラックス定義（実装しやすい版）
1反応 j の時刻 t における寄与を
- `p_k = max(ν[k,j], 0)`（生成側係数）
- `c_i = max(-ν[i,j], 0)`（消費側係数）

として、反応速度 `rop_j(t)` から
- `flux_{i→k}(t; j) = rop_j(t) * c_i * p_k / (sum_i c_i)`  
  （消費側で正規化して、生成側へ配分）
のように割り振る（簡易・安定）。

これを j と t で足して `F(t)` または位相積分 `F̄(phase)` を作る。

### 4.2 位相別（ALD）
ALDでは `pulse1/purge1/pulse2/...` のように位相があり、
同じ反応でも位相で重要性が逆転する。

- `phase_id(t)` を生成し、  
  `F̄_phase = ∫_{phase} w(t) F(t) dt` を作る。
- w(t) は等重みでも良いが、  
  “成膜に効く位相だけ重みを高く” などもconfig化できるようにする。

---

## 5. 重要度（reaction/species importance）
- `I_reaction[j] = Σ_cases Σ_t |rop_j(t)| * dt`（基本）
- `I_species[i] = Σ_cases Σ_t |wdot_i(t)| * dt`（基本）
- 位相別重要度 `I_reaction[j, phase]` も用意（縮退の安定化に効く）

---

## 6. テスト（pytest）
最低限テストすること：
1) ν のshapeが Ns×Nr で、反応式数と一致  
2) 元素保存: `A @ ν ≈ 0`（整数誤差除きゼロ）  
3) フラックス: `F` が非負で、総量が rop/ν 由来のスケールと整合  
4) 位相分割: phase境界が期待通り（ALDベンチマークで固定seed）

---

## 7. CLI（任意だが推奨）
例：  
- `python -m cantera_model.build_dataset --config configs/dataset.yaml`  
- `python -m cantera_model.build_network --dataset path --out path`

