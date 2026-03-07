# Canteraベンチマークを核にした反応ネットワーク縮退の実装計画

## リポジトリ現状の読み取りと役割分担

本件で選択されている2リポジトリは、役割分担が自然に分けられます。

cantera_model は「Canteraでベンチマーク条件を回し、baseline と candidate（縮退機構）を同一条件で比較して合否や誤差統計を出す」目的に最適化された最小パックです。現状の設計は **“縮退機構の評価ハーネス”** として非常に良く、実装計画ではここを **評価の単一真実（single source of truth）**として固定し、上流で生成された候補（削除・マージ・再合成の結果）を受け取って自動評価する立て付けにします。

chem-pred-kit は「設定駆動（YAML合成）・Process単位のCLI・runs/配下に成果物を残す」運用を前提としたML基盤です。反応機構縮退は、複数ケース（温度・圧力・混合比・ALD位相など）を大量に走らせ、学習・探索・比較を反復するため、**ML実験の運用設計（config→run artifacts→比較）**がそのまま効きます。実装計画では chem-pred-kit 側に **縮退アルゴリズム本体と学習パイプライン** を載せ、cantera_model は **評価・回帰テスト** に徹させる構成が、最短で“回る”形になります。

加えて、Cantera自体は「機構の動的な編集・部分機構抽出・YAML書き出し」や、「反応経路解析（反応パス図）」「大機構向けの前処理（preconditioner）」などを備えているため、縮退候補生成・特徴量抽出・高速化の基盤としても使えます。citeturn0search0turn9search8turn23search1

## 共通データスキーマ設計

縮退手法（Top4）を“差し替え可能”にするには、まず **共通データスキーマ**が必要です。ここは最重要で、後工程の自由度を決めます。

### Cantera結果の最小共通集合

ベンチマーク（gri30）であっても、Top4の実装には少なくとも以下が必要です。

- **条件**：温度、圧力、混合比（phi）、時間範囲など（現状CSVで管理）
- **時系列状態**：T(t)、P(t)、X(t)（種モル分率）  
- **反応レート系**：反応の net rates of progress（r_j(t)）および（可能なら）種の net production rates（\dot{x}(t)）
- **機構構造**：化学量論行列（ν、reactant/product stoich coeffs）と反応式文字列、種名・元素組成

Canteraは、反応の net rates of progress を提供し（bulkでは kmol/m^3/s 等）、化学量論係数行列（reactant/product stoich coeffs）も取得できるので、ν と r(t) から \dot{x}(t)=ν r(t) を構成できます。citeturn23search8turn23search0turn23search2

### 保存形式の提案

反応機構縮退では「X(t) だけ」では足りず、r_j(t) や \dot{x}(t) をケース×時刻で持つため、CSVはすぐ破綻します。以下のどちらかを初期実装で選ぶのが現実的です。

- **HDF5（推奨）**：ケース・時刻・種・反応の4次元データを素直に格納でき、部分読み出しも可能（学習で効く）
- **Parquet（補助）**：ケース×時刻を縦長テーブル化して保存（集計・可視化は便利）

Cantera側にも HDF5 と連携する読み書きがあるため、Canteraオブジェクト由来の配列をHDF5へ落とす運用と相性が良いです（h5pyが必要）。citeturn23search5

### 反応ネットワーク（行列/グラフ）への標準化

同じデータから、用途別に次の2種類のネットワーク表現を標準生成することを提案します。

- **構造グラフ（静的）**：species–reactionの二部グラフ、または species graph（共起/隣接）  
  - 入力：ν（reactant/product stoich coeffs）
- **フラックスグラフ（動的/統計）**：時刻ごとのフラックス行列 F(t) と、その位相別・ケース別集約 \bar{F}  
  - 入力：ν と r(t)

Canteraの ReactionPathDiagram は「特定元素のフラックスに基づく反応パス図」を生成できるため、元素別の経路評価（C, H, F, Si 等）を標準化する際に有用です。citeturn9search4

## 実装アーキテクチャの具体化

ここでは、2リポジトリを前提に「どこに何を実装するか」を具体化します。狙いは **“縮退＝アルゴリズムの差し替え”**を成立させることです。

### cantera_model に追加する最小拡張

cantera_model は評価ハーネスとして既に成立しているため、追加は **データ生成（学習用トレース出力）**に絞ります。

- 新規スクリプト例：`run_cantera_trace.py`  
  - 入力：既存の configs/*.yaml と conditions CSV
  - 出力：`data/traces/<run_id>.h5`（時系列・r(t)・νメタ情報）
- 新規設定例：`configs/gri30_small_trace.yaml`  
  - 既存のベースライン設定を継承し、保存する変数（X, r, \dot{x}）とサンプリング間隔を指定

大機構・長時間で剛性が強い場合、Canteraの `AdaptivePreconditioner` を使う余地を残しておくと、データ生成が安定します（大機構向けの設計思想が明示されています）。citeturn23search1turn23search4

### chem-pred-kit に追加する縮退用パッケージ

chem-pred-kit 側に `src/kinetics/`（名称は任意）を新設し、Process駆動で次の単位を作ります。

- `build_cantera_dataset`：cantera_model が吐いた trace を読み、学習・縮退に必要な特徴量を生成して `data/processed/kinetics/` にキャッシュ  
- `train_reducer`：Top4のいずれかを学習/探索して “縮退器” を作る  
- `export_candidate`：縮退器から candidate（YAML機構 or surrogate model）を出力  
- `evaluate_candidate`：cantera_model の evaluator を呼び出して比較指標を保存（回帰試験）

chem-pred-kit は YAML合成（defaults合成）とCLI分離が既にあり、学習・推論のdispatcherもあるので、縮退にも同じ設計を適用できます（設定ファイル1枚で run artifacts を再現できる形）。これは縮退の探索反復（GA/ベイズ最適化/段階縮退）に特に効きます。

### “縮退候補”の共通インターフェース

Top4を同一枠で回すには、candidate を次の2系統に分けた上で共通評価できる設計が必要です。

- **Cantera YAML候補**：削除中心（skeletal）や、一部のマージ（代表種へ写像）で実現できる候補  
- **Surrogate候補**：マージや再合成で“擬似状態（代表状態）”を作る手法（LearnCK系やSINDy→CRN再構成など）

Cantera YAML候補の生成は、Cantera公式の “mechanism reduction” 例が示す通り、反応をランキングして上位を抽出し、必要な種を集めて `ct.Solution(species=..., reactions=...)` を組み直し、YAMLへ書き出す流れがベースラインになります。citeturn0search0turn9search8

一方、Surrogate候補は Canteraの枠外で積分する必要が出るため、評価器側を拡張して「baseline は Cantera」「candidate は surrogate integrator」で同一QoIを計算できるようにします。保存則は ANN-hard のように “出力後に厳密射影する”方式を採ると、縮退で壊れやすい質量・元素保存を実装レベルで担保しやすいです。citeturn11search4turn11search47

## Top4手法セットの実装計画

ここからが本題で、各手法を「この2リポジトリで動く形」に落とします。NeuralODEは使いません（候補生成・source term学習・SINDy等で完結させます）。

### 強い縮退と物理整合を狙う LearnCK++ 系

LearnCKは、巨大な微視的機構から **安定種のみの overall reactions** とその反応速度を学習し、状態数（ODE数）と剛性を大きく落とす狙いの枠組みです。citeturn10search0turn10search1  
半導体向けには「安定種」だけでなく、ご要望の **状態種×元素族×役割**の代表状態（任意状態）を作りたいので、実装計画では次の2段に分けます。

- **代表状態の定義（マージ規則の導入）**  
  - 例：{C,H}族、{C,F}族、{Si,Cl,O}族…  
  - 相・電荷・表面サイトは原則分離（mask）  
  - ラジカル/中性は “跨ぎ許容だがペナルティ” として実装（後段が学習で吸収）
- **overall reaction の再合成と速度学習**  
  - 代表状態間の overall reaction セット（初期は人手、次に自動提案）  
  - 代表状態と運転条件（T, P, 供給、位相ラベル等）から overall rates を回帰

overall rate の表現は、解釈性を保つため CRNN 系を採用します。CRNNは質量作用則・Arrhenius則を埋め込む設計で「反応経路をNN重みとして読める」ことが要点です。citeturn15search4  
さらに、原子保存をNN層で保証する atom-conserving CRNN が報告されており、欠測やノイズへのロバスト性も改善するため、縮退学習に向きます。citeturn16search1

半導体プロセスでは圧力依存や位相（パルス/パージ）の外部変数依存が強いので、CRNNのパラメータを外部変数の関数として学習できる KA-CRNN を “速度則の拡張器” として組み込みます（NeuralODEではなく、パラメータ関数化による拡張）。citeturn20view0

この手法セットの実装成果物（chem-pred-kit 側）は、概ね次になります。

- `models/learnckpp/<run_id>/`：overall reaction 定義、trained model、保存則射影設定
- `exports/learnckpp/<run_id>/`：surrogate runner 用のモデルファイル（Cantera YAMLではなくても可）
- `eval/<run_id>/`：cantera_model を基準にしたQoI誤差、縮退率、失敗ケース分析

### 階層縮退と“自動マージ”を実現する グラフPooling系

マージ（S：種→代表ノード割当）を学習する本命がここです。ただし反応は多反応物・多生成物なので、通常グラフよりハイパーグラフ表現が自然です。実装では、まず **species–reaction 二部グラフ**（反応ノードを導入）として PyTorch Geometric 上のグラフとして扱い、Poolingで species 側を粗視化します。

Poolingの選定は「縮退レベルを固定しない（クラスタ数を自動決定）」が重要で、BN-Poolのようなベイズ非パラメトリックPoolingがこの要件に一致します。citeturn11search1turn12search7  
また SpaPool は “適応的なクラスタ数”を掲げるPoolingで、階層縮退レベルを連続的に探索する際の比較対象になります。citeturn11search3

実装の現実解として、Pooling実験は tgp（Torch Geometric Pool）を採用し、Pooling実装差分を吸収します。tgpは多数のPoolingを統一APIにまとめ、BN-Poolも提供しているため、まずここで高速に当たりを付けられます。citeturn12search2turn12search4turn11search0

マージ制約（半導体観点）の実装は、Poolingに対する **mask（禁止）＋penalty（罰則）**で実現します。

- 禁止（mask）：気相と表面を同クラスタにしない、電荷状態が明確に異なるものは混ぜない等
- 罰則（penalty）：元素族が違うものは強く罰、ラジカル/中性跨ぎは弱く罰  
- 追加：ALD位相（pulse/purge）別に特徴量を作り、位相で挙動が真逆になる種のマージを罰

Pooling後の reduced dynamics は2案を持ちます。

- **案A（解釈性優先）**：CRNNコアで “代表反応” を明示しつつ学習（atom conservation層を併用）citeturn15search4turn16search1  
- **案B（実装容易・高速）**：source term を直接回帰し、ANN-hard射影で保存則を満たすciteturn11search4turn11search47

このセットは “マージを学習する” ため、最初に GRI30 で動かし、つぎに ALD/CVD 表面機構へ移す順が安全です。表面のときは被覆率補正が必須になりやすく、SRNNの考え方（表面被覆率補正・質量保存）を参照して制約項を設計します。citeturn16search2

### 超大規模を回す現実解としての SL/DeePMR → Pooling

Poolingは重いので「まず削ってからマージする」2段構成が実務的です。削除側は、従来のDRGEP等ではなく、データ駆動のSL/DeePMRを採用します（ご要望に一致）。

- SL（sparse learning）は、反応重要度をデータから学習して反応をランキングし、よりコンパクトな縮退機構を作る狙いです。citeturn13search0  
- DeePMR は、縮退を組合せ最適化として扱い、DNNとGAを組み合わせて探索効率を上げる設計です。citeturn13search1

このセットの実装手順は次になります。

- cantera_model の trace 出力から、ケース×時刻の r(t), X(t), \dot{x}(t) を取得
- SL：反応ベクトルのスパース重みを最適化し、反応ランキングを出す（chem-pred-kitで学習プロセス化）citeturn13search0  
- DeePMR：  
  - 候補を「種マスク」として表現  
  - candidate を生成（Canteraの機構編集→YAML書き出し）citeturn0search0turn9search8  
  - cantera_model の evaluator でQoI誤差を計算  
  - その評価値を教師データとして DNN surrogate を更新し、GA探索を加速 citeturn13search1
- 削除で小さくなったネットワークに対して、Pooling（前節）でマージを学習

このセットの成果物は、最終的に **Cantera YAML候補**に落ちやすいのが利点です（現行ベンチと相性が良い）。Canteraは機構の編集・抽出と reduced mechanism の再実行例が公式にあるため、まずここで“確実に回る縮退”を作り、次のセットへ進むのが最短です。citeturn0search0

### 縮退効果の最大化を狙う SINDy拡張とCRN再構成

最後が「縮退効果最大化」のセットです。ここは “元機構を削る/まとめる” を越えて、**データから最小反応網を再発見する**方向を取ります。

中核は、次の2段です。

- ノイズに強い **積分形式による同定**（微分を直接推定しない）
- 同定した方程式を **許容される質量作用則のCRNへ自動復元**（機構として出力可能にする）

2026年2月提出の arXiv 論文は、濃度データから積分形式での同定を用い、その後に **質量作用則に整合する反応ネットワークを自動的に復元**する統一枠組みを提示しています。citeturn21view0  
これは「SINDyで出た式を、反応式として扱える形に戻す」という点で、縮退後の運用（説明・管理・移植）に非常に効きます。

このセットを “複数条件（パラメータスイープ）” に適用するために、次を組み合わせます。

- GS-SINDy：Earth Mover’s Distance と group similarity を使って複数データセットの共通構造を安定に抜き出す提案があります。citeturn14search2turn14search1  
- ADAM-SINDy：未知の非線形パラメータを同時最適化し、候補ライブラリ依存を下げる設計です。citeturn22view0turn15search47  
- 剛性対策：A-stable な陰的Runge–Kuttaを組み合わせるIRK-SINDy系の提案があり、剛性が強い反応系への耐性として参照価値があります。citeturn14search0

評価は、cantera_model のQoI（点評価）に加え、代表状態の時系列誤差（RMSEなど）を導入します。CRN復元後に Cantera YAMLとして表現できる場合は YAML出力して既存 evaluator に流し、難しい場合は surrogate積分で QoI を同一計算し比較します。

## 妥当性評価の自動化とベンチマーク運用

評価は「縮退率の最大化」と「誤差制約」を同時に扱うため、単一スコアではなく、**Paretoフロント**として保存する運用が向きます。

### 既存 evaluator を核にした評価レイヤ

cantera_model の evaluator は baseline/candidate 比較を前提としているため、縮退器側（chem-pred-kit）からは次だけ渡せば回ります。

- candidate mechanism path（Cantera YAML候補の場合）
- conditions CSV（small / netbench_val など）
- 許容相対誤差 `rel_tolerance` など

ここで、評価KPIを次の2階層に分けると運用が安定します。

- **合否（gate）**：pass_rate ≥ 閾値、worst_case の特定、失敗理由（NaN、負濃度、未定義種など）
- **最適化指標**：  
  - 縮退効果：種数、反応数、（マージなら）代表状態数  
  - 精度：平均相対誤差、最大誤差、QoIごとの誤差分布

### 保存則の自動検査

縮退で最も壊れやすいのが保存則なので、評価時に必ずチェックします。

- Cantera YAML候補：元素保存は機構構造上は保たれるが、計算誤差や不安定（負濃度）を検出  
- surrogate候補：ANN-hard のように「予測後に保存則を厳密に満たすよう射影する」ことで、評価の土台を固定できます。citeturn11search4turn11search47

### 反応経路／階層性の“診断”出力

縮退は「できた/できない」だけでなく、なぜ縮退できたか（どの経路が支配的か）をログとして出す必要があります。Canteraの ReactionPathDiagram は、元素別の反応パス情報とdot出力を備えているため、縮退前後で **パス構造がどう変わったか** を自動レポート化する足場になります。citeturn9search4turn9search6

## 半導体ALD/CVD・プラズマを見据えた制約設計と拡張

最後に、GRI30ベンチを“出発点”として、ALD/CVD（表面）やプラズマ（電子温度、電子衝突）へ自然に拡張するための制約方針を具体化します。

### 表面反応の制約

ALD/CVDでは、表面種・サイト密度・被覆率が支配要因になります。Canteraは `Interface` を通じて表面反応と隣接bulk相を扱えるため、表面系へ拡張する際のAPI基盤は既にあります。citeturn9search8  
ただし縮退（特にマージ）では、被覆率補正やサイト収支を崩しやすいので、SRNNが提示する「表面被覆率補正＋質量保存」を制約の基本設計として取り込みます。citeturn16search2

### プラズマ反応の制約

プラズマでは電子温度がガス温度と異なるため、反応速度が2温度に依存する場合があります。Cantera YAMLは `two-temperature-plasma` などの反応タイプを持ち、速度式も明示されています。citeturn17search0turn17search2  
この領域では「反応タイプ（速度則）を保ったまま縮退し、必要な反応だけ残す」設計が現実的で、KA-CRNNのように外部変数（圧力・電子温度・位相）に対して速度パラメータを関数化する枠組みは、経験式（PLOG等）の置換候補になり得ます。citeturn20view0turn17search2

さらに、半導体のPSI（plasma-surface interactions）では、状態遷移を master equation 構造で保持しつつ遷移率のみNNで学習する NME（neural master equation）フレームワークが、NeuralODEより外挿性と物理整合を重視する設計として報告されています。citeturn16search0  
ALD/ALE/RIEのように「位相が明確」「状態が離散的」「遷移が多い」問題では、NMEは“マージした状態集合”の表現と相性が良い可能性があります。

### Canteraへ戻すための実装オプション

最終的にCFD等へ組み込みたい場合、「Cantera YAMLで表現できる」ことが運用上の大きな価値になります。そのために、次の escape hatch を初期から用意します。

- **custom reaction（Python lambda）**：特定の反応だけを custom rate に置換しても ignition 計算を回せる例が示されています（ベンチ用に有効）。citeturn9search2turn9search3  
- **ExtensibleRate**：入力ファイルやfactoryから構築可能なユーザー定義反応速度の拡張点として用意されています（中長期でC++/Py拡張を検討する場合の軸）。citeturn9search3turn9search7

---

この実装計画は、まず **GRI30ベンチ（cantera_model）で縮退探索→自動評価→回帰試験** を確立し、次に **マージ（Pooling/overall reaction/SINDy→CRN）** を増やしていく順で、最小リスクで段階的に“半導体装置の複雑系”へ拡張できるように設計しています。Cantera本体も近年のリリースで機能拡張が継続しているため、依存バージョン方針（cantera_modelは現状3.1固定）をどこで更新するかも、評価の再現性と合わせて運用ルール化するのが安全です。citeturn6search1turn6search3