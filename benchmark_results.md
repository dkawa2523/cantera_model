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

## 3. 手法ごとの制約
| 手法 | 制約（主） |
|---|---|
| `baseline` | element-overlap hard ban, phase/site hard mask, physical gate, Hard Band(dynamic), floors |
| `learnckpp` | overall候補再合成 + sparse選択 + coverage postselect + projection + Hard Band |
| `pooling` | graph-based assignment + hard mask + cluster guard + learnckpp bridge |

共通必須:
- `hard_ban_violations=0`
- 保存則違反/負濃度 gate
- `pass_rate >= 0.75` かつ `mean_rel_diff <= 0.40`

## 4. QoL適用内容（Qol_new準拠）
- Physics-first（元素保存・サイト整合・非負）
- Hard Band + dynamic bounded
- Gate-First + Compress
- split: `adaptive_kfold`
- QoI: `species_last/species_max` + `species_integral/deposition_integral`

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

## 8. 総合考察
1. diamondは要求どおり「40+種/100+反応」へ置換できた（floor適用により stage=B採択）。
2. 3ベンチとも物理制約（hard-ban/保存則/非負）は維持できている。
3. 不合格の主因は物理破綻ではなく、`adaptive_kfold + integral QoI` 下での誤差（pass_rate不足）。
4. ac/diamondはfold間の誤差分散が大きく、特定foldで大きく崩れる。
5. sif4 pooling full-caseは実行停滞が残課題（計算経路の安定化が必要）。

## 9. 次改善アクション
1. `sif4 pooling(full-case)` の停滞解消（backend切替・stage分割実行・タイムアウト制御）。
2. kfoldで崩れるfoldをケース単位で分解し、積分QoI寄与（`X_int:*`, `dep_int:*`）の重み再設計。
3. learnckpp/poolingでも反応ドメイン分解を出せるよう、overall候補にドメイン由来タグを保持する。
4. gate判定に worst-fold 指標を正式採用し、平均値依存を減らす。
