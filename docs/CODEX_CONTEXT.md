# Codex Context: Cantera反応ネットワーク縮退（削除＋マージ＋代表状態）実装パケット

## 0. 目的（このリポジトリで実装すること）
半導体製造装置（ALD/CVD/プラズマ含む）での巨大反応機構を対象に、以下を **反応ネットワーク（行列/グラフ）前提**で自動化する。

1. **反応ネットワーク構築**  
   - Cantera出力（時系列の濃度・反応レート・生成速度など）と、機構（species/reactions）から  
     - 化学量論行列 ν（species×reactions）
     - フラックス行列（species→species）や反応重要度（reaction importance）
     - 位相別（ALDのpulse/purge等）統計量  
     を生成できるようにする。

2. **縮退（reduction）**  
   - **削除（prune）**：重要でない反応/種の候補を機械学習で同定（DRG/DRGEP依存ではない）
   - **マージ（merge / lump）**：同組成に限定せず、  
     - 状態種（neutral/radical/ion/surface）  
     - 元素族（例: {C,H}, {C,F}, {Si,Cl,O}…）  
     - プロセス役割（成膜寄与/エッチ寄与/キャリア/壁吸着…）  
     を考慮し、代表ノード（任意状態）へ集約する
   - **代表反応（代表系の反応式）再合成**：マージ後の機構（状態数・反応式数が少ない機構）を生成する

3. **妥当性評価（validation）**  
   - 複数のCantera実行ケース（条件スイープ）で  
     - 時系列誤差
     - KPI（ALDならGPC/飽和時間/副生成物、CVDなら成膜速度/副生成物、プラズマならラジカル密度など）  
     - 保存則（元素・電荷・表面サイト）  
     を自動評価し、縮退の可否と縮退レベル（弱/中/強）をレポート化する。

## 1. 前提（NeuralODEは使わない）
- 微分方程式を「ニューラルODEとして学習」しない（torchdiffeq等は使わない）。
- 学習は主に「(x(t),条件)→dx/dt」や「(y(t),条件)→代表反応速度」などの教師あり/自己教師ありで行い、
  縮退機構の**検証**でCantera/ODE積分を行う。

## 2. Top4手法（この実装で狙う4本）
A. LearnCK++ / CRNN / SRNN / ANN-hard（強い縮退＋物理整合）  
B. 反応ネットワークPoolingでS（種→代表ノード割当）を学習（階層縮退の本命）  
C. 反応重要度を学習して剪定→Poolingでマージ（超大規模向け現実解）  
D. SINDy系（Integral/weak + group sparsity）＋CRN再構成（縮退効果最大化・発見型）

## 3. 実装時に守る設計指針
- **データ契約（schema）を先に固定**：以降の手法は同一のDataset APIを使う
- **疎行列を基本**：ν, incidence, flux などは scipy.sparse / torch sparse を基本にする
- **再現性**：seed固定、設定はYAML/JSONで保存、実験ログはファイル出力
- **テスト**：ν作成、フラックス集約、保存則チェックはpytestでユニットテスト化

## 4. 依存ライブラリ（推奨セット）
- 数値/データ: numpy, scipy, pandas, xarray, zarr(or h5py), pyarrow(parquet)
- Cantera連携: cantera (>=3.0系), ruamel.yaml
- グラフ: networkx(可視化/小規模), torch, torch_geometric, tgp(torch-geometric-pool)
- 学習/最適化: pytorch-lightning(or plain torch), optuna, scikit-learn
- 制約/凸最適化: cvxpy, osqp(or scs)
- SINDy: pysindy（必要に応じて自前実装）
- 実験管理: wandb or mlflow（どちらか一つでOK）
- CLI: typer, rich

## 5. リポジトリを読む順番（Codexが最初に確認すべき）
1) README / quickstart  
2) Cantera実行コード（複数条件実行の入口）  
3) ベンチマークケースの置き場所と実行手順  
4) 出力（ログ/CSV/npz等）の形式  
5) 既存の解析/可視化コードがあるか

---

## 付録：用語
- ν: 化学量論行列（species×reactions）
- r_j(t): rate-of-progress（反応進行速度）
- ω̇_i(t): species net production rate
- F(t): species→speciesのフラックス（反応寄与を集約したネットワーク）
- P or S: 種→代表ノード（cluster）への写像行列
