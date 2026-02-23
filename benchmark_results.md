# Benchmark Results (Large 3-Benchmark, Eval9)

更新日: 2026-02-23

## 1. 目的
large系3ベンチマーク（`benchmarks_diamond` / `benchmark_sif4_sin3n4_cvd` / `benchmark_large`）を再実行し、
以下を同時に評価した。

- 物理妥当性（保存則・非負・hard-ban）
- 縮退効果（species / reactions）
- `adaptive_kfold` での汎化
- 積分QoI（deposition/byproduct/surface）を含む gate 判定
- 気相反応・表面反応を含む縮退挙動

## 2. ベンチマーク定義（今回の実行）
| benchmark | ケース | trace cases | species | reactions | 備考 |
|---|---|---:|---:|---:|---|
| `benchmarks_diamond` | `diamond_cvd_large` | 4 | 78 | 385 | **diamondは40+/100+条件に合わせて採択床を更新** |
| `benchmark_sif4_sin3n4_cvd` | `sif4_sin3n4_cvd_large` | 6 | 82 | 365 | full-case trace |
| `benchmark_large` | `ac_hydrocarbon_cvd_large` | 8 | 94 | 425 | full-case trace |

参照trace:
- `artifacts/traces/diamond_benchmarks_diamond_large_trace_eval6.h5`
- `artifacts/traces/sif4_benchmark_large_trace.h5`
- `artifacts/traces/ac_benchmark_large_trace.h5`

## 3. QoL適用内容（Qol_new準拠）
- Physics-first（元素保存・サイト整合・非負）
- Hard Band + dynamic bounded
- Gate-First + Compress
- split: `adaptive_kfold`
- QoI: `species_last/species_max` + `species_integral/deposition_integral`

## 4. 手法ごとの制約（現行実装）
### 4.1 全手法共通の必須制約
- `hard_ban_violations=0`（共通元素ゼロのマージ禁止を厳守）
- 物理 gate:
  - 元素保存違反が閾値以下
  - 負濃度ステップ数が閾値以下（実運用では0）
- QoI gate:
  - `pass_rate >= 0.75`
  - `mean_rel_diff <= 0.40`
- split:
  - `adaptive_kfold` を既定
  - `n_cases<4` のみ `in_sample` へフォールバック

### 4.2 手法別の制約詳細
| 手法 | 構造制約（Hard） | 学習/選択制約（Soft） | 評価時の注意点 |
|---|---|---|---|
| `baseline` | element-overlap hard ban、phase/site hard mask、physics floors | stage A/B/C の `target_ratio`, `prune_keep_ratio`, `penalty_scale`, dynamic Hard Band | 反応は元機構反応を直接 keep/drop するため、gas/surface の反応削減内訳が直接読める |
| `learnckpp` | `S` 由来のクラスタ空間で保存則射影、fallback to baseline | `overall_candidates` 生成、`target_keep_ratio`, `min_keep_count`, coverage-aware postselect | `overall` 再合成反応を使うため、反応の gas/surface 1対1分解は未定義になりやすい（`reaction_domain_split_available=false`） |
| `pooling` | graph assignment + hard mask（element/phase/site）、cluster guard | `min_clusters`, `coverage_target`, `max_cluster_size_ratio`, train backend（pyg/numpy） | 計算コスト・収束性が backend 依存。large + full-case で停滞リスクがある |

### 4.3 benchmark別に効いている制約
| benchmark | 支配的な制約 | 実務上の意味 |
|---|---|---|
| diamond_large | `min_species_abs=40`, `min_reactions_abs=100`, `max_reaction_species_ratio` | 過圧縮を抑制し、反応ネットワークの可読性を維持 |
| sif4_large | kfold汎化 + 積分QoI + Hard Band | 反応数が少ない表面反応でも、積分KPIで見て誤差が拡大しやすい |
| ac_large | kfold汎化（fold間分散） + dynamic Hard Band | 条件差が大きく、特定foldで誤差が急増しやすい |

### 4.4 フォールバック/失敗時ポリシー
- `learnckpp` 失敗時: baseline経路へフォールバック可能（設定依存）
- `pooling` 失敗時: rule-based mergeへのフォールバック、または backend 切替（`pyg -> numpy`）
- split失敗時: `kfold -> in_sample` に自動切替し、`summary.surrogate_split.fallback_reason` に理由記録
- 採択不能時: gate未通過の中でも `pass_first_pareto` で最も制約順守側を選択し、失敗理由を `gate_evidence` に残す

## 5. diamond置換内容（今回の要求反映）
`configs/reduce_diamond_benchmarks_large_*.yaml` を更新:
- `physics_floors.min_species_abs: 40`
- `physics_floors.min_reactions_abs: 100`

これにより、diamondの縮退採択は「40+種 / 100+反応」未満を不採択にした。

## 6. 評価結果（Eval9）

### 6.1 benchmarks_diamond (large, 40+/100+ floor適用)
参照:
- `reports/eval9_diamond_large_baseline/summary.json`
- `reports/eval9_diamond_large_learnckpp/summary.json`
- `reports/eval9_diamond_large_pooling/summary.json`

| mode | gate | stage | species_after | reactions_after | pass_rate | mean_rel_diff | split |
|---|---|---|---:|---:|---:|---:|---|
| baseline | false | B | 43 | 270 | 0.00 | 4.6897 | kfold(2/4) |
| learnckpp | false | B | 43 | 100 | 0.00 | 4.2078 | kfold(2/4) |
| pooling | false | B | 40 | 100 | 0.00 | 4.2078 | kfold(2/4) |

### 6.2 benchmark_sif4_sin3n4_cvd (large)
参照:
- `reports/eval9_sif4_large_baseline/summary.json`
- `reports/eval9_sif4_large_learnckpp/summary.json`
- `reports/eval8_sif4_large_pooling/summary.json`（注: poolingは6-case runが停滞し、直近完了runを採用）

| mode | gate | stage | species_after | reactions_after | pass_rate | mean_rel_diff | split |
|---|---|---|---:|---:|---:|---:|---|
| baseline | false | C | 29 | 182 | 0.1667 | 0.2791 | kfold(3/6) |
| learnckpp | false | C | 28 | 12 | 0.1667 | 0.3220 | kfold(3/6) |
| pooling* | false | A | 21 | 11 | 0.0000 | 0.2980 | kfold(2/4) |

### 6.3 benchmark_large (ac_large)
参照:
- `reports/eval9_ac_large_baseline/summary.json`
- `reports/eval9_ac_large_learnckpp/summary.json`
- `reports/eval9_ac_large_pooling/summary.json`

| mode | gate | stage | species_after | reactions_after | pass_rate | mean_rel_diff | split |
|---|---|---|---:|---:|---:|---:|---|
| baseline | false | C | 33 | 212 | 0.0000 | 7.3213 | kfold(3/8) |
| learnckpp | false | C | 33 | 26 | 0.0000 | 7.0618 | kfold(3/8) |
| pooling | false | C | 8 | 13 | 0.0000 | 7.0618 | kfold(3/8) |

## 7. 気相/表面縮退の評価

### 7.1 baseline（直接反応選択なので gas/surface 分解可能）
| benchmark | gas reactions before->after | surface reactions before->after | gas species before->after | surface species before->after |
|---|---:|---:|---:|---:|
| diamond | 325 -> 227 | 60 -> 43 | 53 -> 27 | 25 -> 16 |
| sif4 | 347 -> 164 | 18 -> 18 | 62 -> 21 | 20 -> 8 |
| ac | 325 -> 148 | 100 -> 64 | 53 -> 27 | 41 -> 6 |

### 7.2 learnckpp / pooling（注意）
- これらは overall再合成反応を使うため、反応の gas/surface 1対1分解は `summary` 上で未定義（`reaction_domain_split_available=false`）。
- ただし状態クラスタは gas/surface に分けて追跡可能。

## 8. 総合考察（詳細）
### 8.1 まず成立している点
1. 3ベンチすべてで、物理制約のコアは維持できている。具体的には `hard_ban_violations=0`、保存則/非負の gate は通過しており、「物理的に壊れたモデル」は排除できている。
2. diamond は要求どおり `40+ species / 100+ reactions` の採択条件に置換できた。過圧縮を抑えた状態で比較可能なラインに揃っている。
3. trace を full-case（sif4:6, ac:8）へ戻したことで、`in_sample` では見えない汎化失敗が検出できる評価系になった。これは短期的には不合格増だが、運用上は正しい方向。

### 8.2 gate 不合格の構造的原因
1. 主因は物理違反ではなく、`adaptive_kfold + integral QoI` 下の予測誤差拡大である。`pass_rate` が閾値 0.75 に届かず、`mean_rel_diff` も fold により大きく悪化する。
2. diamond/ac では fold間分散が大きく、ある fold だけ誤差が突出して全体不合格を作る。平均値では見えにくい「最悪fold主導」の失敗モードが明確。
3. sif4 は平均誤差は比較的小さいが、積分QoIを含めると閾値近傍の QoI が落ちやすく、`pass_rate` を押し下げる。つまり「少数の難QoI」が全体合否を支配している。

### 8.3 手法別の考察
1. `baseline`:
  - 反応の gas/surface 内訳を直接追跡でき、診断性が最も高い。
  - ただし圧縮を進めると特定foldで誤差急増しやすく、kfold下で安定しにくい。
2. `learnckpp`:
  - 反応数圧縮は強いが、overall再合成反応のためドメイン別反応内訳の可視性が下がる。
  - 物理整合は維持しても、fold外推で QoI誤差が増えると gate を越えにくい。
3. `pooling`:
  - 状態圧縮は最も強く、species削減効果が大きい。
  - 一方で large/full-case 条件で計算停滞が出やすく、評価完走性がボトルネック。現状は「精度以前に実行安定性」の改善が必要な局面。

### 8.4 benchmark別の考察
1. diamond_large:
  - 40+/100+ 制約で「意味の薄い極端圧縮」は抑止できた。
  - ただし QoI誤差が依然大きく、構造妥当性だけでは gate 合格に至らない。
2. sif4_large:
  - baseline/learnckpp は full-case(k=3) で再評価できたが不合格。
  - pooling の full-case は停滞し、運用判断に必要な比較が未完。現時点では「評価系の完走性」が最優先課題。
3. ac_large:
  - fold依存の外れが強く、平均より worst fold が支配。
  - pooling は強圧縮できるが、現状は精度面で基準未達。

### 8.5 制約設計の妥当性評価
1. 今回の制約は「過圧縮を防ぎつつ物理整合を守る」点では機能している。
2. 一方で、制約を守っても QoI汎化が追従していないため、次段階は「制約緩和」ではなく「誤差源の分解と学習/選択の再重み付け」が中心になる。
3. 特に積分QoIは運用KPIに近い反面、評価難易度を上げる。したがって QoIごとの重要度とfold寄与を分解し、重みと gate 判定の設計を再調整する必要がある。

## 9. 次改善アクション
1. `sif4 pooling(full-case)` の停滞解消（backend切替・stage分割実行・タイムアウト制御）。
2. kfoldで崩れるfoldをケース単位で分解し、積分QoI寄与（`X_int:*`, `dep_int:*`）の重み再設計。
3. learnckpp/poolingでも反応ドメイン分解を出せるよう、overall候補にドメイン由来タグを保持する。
4. gate判定に worst-fold 指標を正式採用し、平均値依存を減らす。
