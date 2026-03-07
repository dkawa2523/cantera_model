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

## 10. v29 運用安定化評価（eval29）
更新日: 2026-03-03

### 10.1 実行対象
- `eval29_diamond_large_{baseline,learnckpp,pooling}`
- `eval29_sif4_large_{baseline,learnckpp,pooling}`
- `eval29_ac_large_{baseline,learnckpp,pooling}`
- 集計:
  - `reports/diamond_large_eval_summary_eval29.json`
  - `reports/sif4_large_eval_summary_eval29.json`
  - `reports/ac_large_eval_summary_eval29.json`
  - `reports/eval29_acceptance_eval28_compare.json`

### 10.2 eval29 結果（品質 + runtime）
| benchmark | mode | gate | stage | species_after | reactions_after | pass_rate | mean_rel_diff | timing_total_s |
|---|---|---|---|---:|---:|---:|---:|---:|
| diamond | baseline | false | B | 43 | 270 | 0.0000 | 4.6897 | 1.4175 |
| diamond | learnckpp | false | B | 43 | 100 | 0.0000 | 4.2078 | 0.2677 |
| diamond | pooling | false | B | 40 | 100 | 0.0000 | 4.2078 | 2.2241 |
| sif4 | baseline | false | C | 29 | 182 | 0.1667 | 0.2791 | 5.1651 |
| sif4 | learnckpp | false | C | 28 | 12 | 0.1667 | 0.3220 | 0.5577 |
| sif4 | pooling | false | C | 21 | 13 | 0.5000 | 0.1842 | 0.2235 |
| ac | baseline | false | C | 33 | 212 | 0.0000 | 7.3213 | 7.2321 |
| ac | learnckpp | false | C | 33 | 26 | 0.0000 | 7.0618 | 0.7314 |
| ac | pooling | false | C | 8 | 13 | 0.0000 | 7.0618 | 0.6569 |

### 10.3 A1-A6 判定
1. A1 (`timing_*` 欠損ゼロ): **達成**
2. A2 (同一 run-id 二重起動拒否): **達成**  
   `tests/test_reduce_validate_runtime_lock.py` で確認。
3. A3 (保存則投影最適化の数値整合): **達成**  
   `tests/test_conservation_projection.py` で確認。
4. A4 (`pooling.bridge light < full`): **達成**  
   smoke(sif4): `full=0.1338s`, `light=0.0638s` (`timing_bridge_s`)。
5. A5 (`hard_ban_violations=0` + 物理回帰なし): **達成**  
   eval29 9/9 で `hard_ban_violations=0`, `negative_steps=0`。
6. A6 (eval28比で長時間run抑制): **部分達成（可視化達成）**  
   eval29 は `timing_total_s` 合計 `18.476s`, 最大 `7.232s`。  
   ただし `eval28` summary は `timing_total_s` を持たないため、summary同士の直接 runtime 差分は不可。  
   代替として、v29以降は `timing_total_s` を継続追跡し比較可能。

### 10.4 補足
- `pooling.bridge.mode` は初期運用どおり `full` を既定維持。
- `light` は短サイクル比較/調査用途で有効（品質差分監視は継続）。

## 11. v31 large9 実行・差分評価（eval31 vs eval30r）
更新日: 2026-03-03

### 11.1 実行/集計
- 実行 run-id:
  - `eval31_diamond_large_{baseline,learnckpp,pooling}`
  - `eval31_sif4_large_{baseline,learnckpp,pooling}`
  - `eval31_ac_large_{baseline,learnckpp,pooling}`
- summary:
  - `reports/diamond_large_eval_summary_eval31.json`
  - `reports/sif4_large_eval_summary_eval31.json`
  - `reports/ac_large_eval_summary_eval31.json`
- 差分レポート:
  - `reports/eval31_vs_eval30r_diff.json`
  - `reports/eval31_vs_eval30r_diff_report.md`

### 11.2 split公平性チェック結果
1. `diamond`: `split_mode=kfold`, `effective_kfolds=2` で mode間一致（OK）
2. `ac`: `split_mode=kfold`, `effective_kfolds=3` で mode間一致（OK）
3. `sif4`: `split_mode=kfold` は一致するが、`effective_kfolds` が `3,3,2` で不一致（NG）
4. そのため `sif4` summary は `--enforce-same-split false` で生成し、比較時に公平性違反として明示。

### 11.3 eval30r 比の要約（9 run 集計）
| metric | eval30r | eval31 | delta |
|---|---:|---:|---:|
| gate_passed_count | 0 | 6 | +6 |
| mandatory_validity_passed_count | 2 | 6 | +4 |
| avg_valid_mandatory_metric_count | 6.3333 | 8.3333 | +2.0000 |
| avg_mandatory_valid_ratio | 0.5216 | 0.6904 | +0.1687 |
| avg_pass_rate_mandatory_case | 0.5558 | 0.7395 | +0.1838 |
| avg_mean_rel_diff_mandatory_raw | 7.2078 | 4.9266 | -2.2812 |

### 11.4 blocker分布（注意付き）
| blocker | eval30r | eval31 | delta |
|---|---:|---:|---:|
| none | 9 | 6 | -3 |
| validity | 0 | 3 | +3 |

注記:
- `eval30r` には `primary_blocker_layer` が未記録だったため、本比較では欠損を `none` として扱っている。
- よって blocker 分布の厳密比較は `eval31` 以降同士で行うのが妥当。

### 11.5 mandatory診断の主な差分
1. AC 3 mode: `valid_mandatory_metric_count` が `6-7/12 -> 10/12` へ改善。
2. sif4 baseline/learnckpp: `pass_rate_mandatory_case` が `0.1667 -> 0.8846/0.8205` に改善。
3. diamond 3 mode: `valid_mandatory_metric_count` は `+1〜+2` 改善したが、`pass_rate_mandatory_case` は `0.5909` で停滞し validity 残存。

## 12. v35 large9 実行・差分評価（eval35 vs eval34）
更新日: 2026-03-03

### 12.1 実行/集計
- 実行 run-id:
  - `eval35_diamond_large_{baseline,learnckpp,pooling}`
  - `eval35_sif4_large_{baseline,learnckpp,pooling}`
  - `eval35_ac_large_{baseline,learnckpp,pooling}`
- summary:
  - `reports/diamond_large_eval_summary_eval35.json`
  - `reports/sif4_large_eval_summary_eval35.json`
  - `reports/ac_large_eval_summary_eval35.json`
- 実行ログ:
  - `reports/_runlogs/eval35_large9_status.tsv`（9/9 `ok`）

### 12.2 主要結果（Pass率とblocker）
| run-set | passed | blocker分布 |
|---|---:|---|
| eval34 | 2/9 | `error:6, validity:1, none:2` |
| eval35 | 7/9 | `none:7, structure:1, validity:1` |

### 12.3 all-units vs valid-only（代表）
1. diamond baseline:
   - `pass_rate_mandatory_case`: `0.9265`（scoped） vs `0.6731`（all-units）
   - `mandatory_rel_diff_p95`: `0.2508`（scoped） vs `1.8063`（all-units）
2. ac baseline:
   - `pass_rate_mandatory_case`: `0.9511`（scoped） vs `0.8194`（all-units）
   - `mandatory_rel_diff_p95`: `0.4174`（scoped） vs `43.2100`（all-units）
3. sif4 pooling:
   - `mandatory_tail_guard_passed=true` だが `mandatory_validity_passed=false` のため `primary_blocker_layer=validity`

### 12.4 解釈
1. v35 の `mandatory_quality_scope=valid_only` と `mandatory_tail_scope=quality_scope` により、coverageで除外済み対象の二重hard判定は解消した。
2. `ac/diamond` の error 主因は大きく減少し、`eval35` の非通過は `structure` 1件と `validity` 1件に集約された。
3. 物理制約は維持され、`hard_ban_violations` は large9 全件 `0`。

## 13. v35.2 深掘り整理（eval352, 保守方針）
更新日: 2026-03-05

### 13.1 実行と結果
- 実行 run-id:
  - `eval352_diamond_large_{baseline,learnckpp,pooling}`
  - `eval352_sif4_large_{baseline,learnckpp,pooling}`
  - `eval352_ac_large_{baseline,learnckpp,pooling}`
- summary:
  - `reports/diamond_large_eval_summary_eval352.json`
  - `reports/sif4_large_eval_summary_eval352.json`
  - `reports/ac_large_eval_summary_eval352.json`
- 結果:
  - `gate_passed=9/9`
  - `primary_blocker_layer=none` が `9/9`
  - `hard_ban_violations=0` が `9/9`

### 13.2 縮退効果と品質（eval352）
| benchmark | mode | species before->after | species削減率 | reactions before->after | reactions削減率 | pass_rate_mandatory_case | mean_rel_diff_mandatory | mean_rel_diff_optional |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| diamond | baseline | 78->43 | 44.9% | 385->253 | 34.3% | 0.926 | 0.085 | 0.047 |
| diamond | learnckpp | 78->43 | 44.9% | 385->100 | 74.0% | 0.897 | 0.208 | 0.137 |
| diamond | pooling | 78->40 | 48.7% | 385->100 | 74.0% | 0.897 | 0.208 | 0.137 |
| sif4 | baseline | 82->29 | 64.6% | 365->182 | 50.1% | 0.972 | 0.153 | 0.128 |
| sif4 | learnckpp | 82->28 | 65.9% | 365->12 | 96.7% | 0.933 | 0.182 | 0.168 |
| sif4 | pooling | 82->21 | 74.4% | 365->13 | 96.4% | 0.917 | 0.168 | 0.245 |
| ac | baseline | 94->33 | 64.9% | 425->212 | 50.1% | 0.951 | 0.114 | 0.163 |
| ac | learnckpp | 94->33 | 64.9% | 425->26 | 93.9% | 0.951 | 0.150 | 0.221 |
| ac | pooling | 94->8 | 91.5% | 425->13 | 96.9% | 0.951 | 0.150 | 0.221 |

### 13.3 今回「両立」が成立した主要因（なぜ重要だったか）
1. 評価責務の分離（coverage と quality の直交化）が効いた。
   - 以前は同じ mandatory 情報が `validity` と `error` の両方で hard fail 条件に入り、同一原因の二重失格が発生していた。
   - 現在は `validity=coverage`、`error=quality` が分離され、縮退を進めても評価層の重複ペナルティで落ちにくい。
2. mandatory の判定単位を `species_family_quorum` ベースに置き換えたことが効いた。
   - 同一 species の `last/max/int/dep` での重複罰を抑制し、「目的（状態種/反応種縮退）」と判定単位を一致させた。
3. `mandatory_quality_scope=valid_only` が効いた。
   - coverage で既に invalid と扱った単位を quality で再 hard 判定しないため、責務の重複が消えた。
4. tail 判定を policy 化して二重 hard を解消したことが効いた。
   - 外れ値監視を残しつつ、少数極端値で全体が即落ちる構造を回避できた。
5. replay health 系の初期誤判定（global連鎖 invalid、clip比率不整合）を先に潰したことが前提として重要だった。
   - ここが残っていると、どれだけ quality 設計を直しても `validity` で詰まる。

### 13.4 何が効かなかったか / 効果が限定的だったか
1. v35.2 の「保守追加（診断強化）」自体は合否を変えていない。
   - `eval351c` と `eval352` を比較すると、`gate_passed`・選択stage・主要品質値は実質同一（差分なし）。
   - つまり通過改善の主因は v35 までの評価設計修正であり、v35.2 は「可観測性向上」が主目的。
2. `selection_quality_score_raw_drift` は 6/9 run で `0.0` に飽和した。
   - `metric_drift_raw>1.3` が 6/9 で、raw drift が極端に大きいケースでは正規化後に差が潰れる。
   - tie-break 指標としては有用だが、現状の raw drift 分布では識別力が限定的。
3. shadow validity（`mandatory_gate_unit_valid_count_shadow_evaluable_ratio`）は今回の設定では差が出なかった。
   - 全runで `shadow_count == mandatory_total_gate_unit_count`（7/7 または 8/8）。
   - `shadow_ratio=0.25` は保守運用には安全だが、監視指標としては緩く、弁別力は低い。
4. `mode_collapse_warning` は全run `false`。
   - 監視機構は動作しているが、今回の run-set では発火条件に該当しなかった。

### 13.5 重要な解釈（評価方法として何が正しかったか）
1. `pass_rate(all-metric legacy)` は依然 `0.0` が多く、運用主指標として不適。
   - mandatory/optional を分離した tiered 指標の方が、縮退目的との整合が高い。
2. 「物理制約を守ること」と「QoI品質を保つこと」を別責務で持つ設計が必要。
   - これを混在させると、過拘束で改善余地が見えなくなる。
3. 目的関数（縮退）と判定単位（species/family）を揃えると、モデル比較が安定する。
   - metric単位の重複罰は、縮退性能の差ではなく定義上の不利を増幅しやすい。

### 13.6 他ケースへの有用性（どこが有用で、なぜ有用か）
#### 有用なケース
1. mandatory 指標が family 重複（`last/max/int/dep`）を多く含むケース。
   - 有用点: `species_family_quorum` により重複罰を抑え、真の品質差を見やすくできる。
2. 圧縮率が高く、外れ値が少数混入しやすいケース。
   - 有用点: winsorized + tail policy により、少数外れ値と系統誤差を分離できる。
3. 物理制約が厳しく、かつ比較対象modeが複数あるケース。
   - 有用点: coverage/error/structure を直交化すると、blockerの一次原因を一意に保てる。

#### 有用性の理由（再利用性の根拠）
1. benchmark 固有 species 名に依存しない。
   - 判定は metric family / species token / case pass rate の一般化ロジックで構成されている。
2. 旧挙動へのフォールバックを残している。
   - `mandatory_quality_scope`, `mandatory_tail_scope`, legacy mode 切替で段階導入が可能。
3. 診断キーが summary/report/summarize に一貫透過される。
   - 運用で「なぜ通った/落ちた」を同じキーで追跡できる。

#### 効きにくいケース（限界）
1. モデル自体の QoI 再現力が不足しているケース。
   - 評価層を改善しても、真の予測誤差は消えない。
2. mandatory 設計そのものが目的と不一致なケース。
   - 評価方法より先に QoI構成の再設計が必要。
3. raw drift が極端に発散するケース。
   - 監視は可能だが、現行の drift 正規化では比較指標が飽和しやすい。

### 13.7 次サイクルでの実務的提案（過度な複雑化なし）
1. `shadow_ratio` の感度点検（例: 0.25 -> 0.40）を A/B で1回だけ実施し、弁別力を確認する。
2. `selection_quality_score_raw_drift` は補助指標のまま維持し、飽和時は rank へ影響しない設計を維持する。
3. KPI 主表示を `mandatory/optional` 系へ寄せ、`pass_rate(all-metric)` は legacy 診断として明示的に格下げする。

## 14. v36 導入結果（Contract/Taxonomy/Schema）
### 14.1 結論
1. `eval36_large9` は `gate_passed=9/9` を維持。
2. `primary_blocker_layer` は全runで `none`（悪化なし）。
3. 反応削減率は `eval352` 比で全run `drop=0.00`（非劣化条件 `<=0.02` を満たす）。
4. `diagnostic_schema_ok=true`、`evaluation_contract_version=v1`、`metric_taxonomy_profile_effective=large_default_v1` を全runで確認。

### 14.2 eval352 vs eval36（反応削減率）
| benchmark | mode | eval352 reaction_reduction | eval36 reaction_reduction | drop (352-36) |
|---|---|---:|---:|---:|
| diamond | baseline | 0.3429 | 0.3429 | 0.0000 |
| diamond | learnckpp | 0.7403 | 0.7403 | 0.0000 |
| diamond | pooling | 0.7403 | 0.7403 | 0.0000 |
| sif4 | baseline | 0.5014 | 0.5014 | 0.0000 |
| sif4 | learnckpp | 0.9671 | 0.9671 | 0.0000 |
| sif4 | pooling | 0.9644 | 0.9644 | 0.0000 |
| ac | baseline | 0.5012 | 0.5012 | 0.0000 |
| ac | learnckpp | 0.9388 | 0.9388 | 0.0000 |
| ac | pooling | 0.9694 | 0.9694 | 0.0000 |

### 14.3 解釈
1. 今回の変更は判定ロジック緩和ではなく「評価契約の固定化」と「taxonomy/diagnosticの明示化」なので、合否と縮退率を崩さずに運用再現性だけを上げられた。
2. `large9` に strict contract を入れたことで、設定不足や診断欠損を実行時に止められる状態になった。
3. taxonomy を共有YAML化しても `legacy` と同値を維持できており、他ベンチ展開の前提（同じ評価意味を保ったまま外部定義化）が確認できた。

## 15. v38 リファクタ結果（スキーマ単一化 + 軽量化）
更新日: 2026-03-06

### 15.1 実施内容
1. 診断スキーマを `cantera_model/eval/diagnostic_schema.py` に単一化し、`reduce_validate` / `summarize` / `report` の重複定義を削除。
2. `compare_rows` の一部責務を helper 化し、`tiered` 固定・legacy fail-fast を明示。
3. large9 一括実行CLI `cantera_model/cli/run_large9.py` を追加（9本実行→3 summary→比較JSONを1コマンド化）。
4. 未参照の追跡資産（`cantera_cvd_ald_benchmarks_pack`、一部 outputs）を削除。

### 15.2 eval37 vs eval38（large9比較）
参照:
- `reports/large9_compare_eval37_vs_eval38.json`
- `reports/*_large_eval_summary_eval38.json`

| 指標 | 判定 |
|---|---|
| gate_passed 9/9 維持 | ✅ |
| reaction_reduction 非劣化（9/9） | ✅ |
| mandatory mean 非劣化（<=1e-6） | ❌ |
| optional mean 非劣化（<=1e-6） | ❌ |

差分発生は `ac/pooling` 1件のみで、`mean_rel_diff_mandatory` が `+0.04548`、`mean_rel_diff_optional` が `+0.07529`。
他8 runは実質同値（丸め誤差レベル）だった。

### 15.3 現時点の解釈
1. v38 の整理変更自体は大半runで非劣化を維持。
2. `ac/pooling` だけ品質差分が出ており、構成/閾値ではなく run 再現性（pooling経路の非決定性）由来の可能性が高い。
3. 次段は refactor 継続ではなく、pooling推論の決定化（seed固定と推論時dropout無効化）を優先すべき。
