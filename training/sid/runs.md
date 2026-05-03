# Run results

| Run | Model | Features | CV | split | rmse_log | rmse_ms | rmse_percent | median_relative_error | p90_relative_error | p95_relative_error | median_percent_error | p90_percent_error | p95_percent_error | median_ratio_error | p90_ratio_error | within_10pct | within_25pct | within_50pct | within_2x |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run One | XGBoost | core features | 0.1 | test | 0.282506 | 132.612615 | 36.069926 | 0.078311 | 0.414712 | 0.590687 | 7.831146 | 41.471205 | 59.068737 | 1.081896 | 1.523052 | 0.572499 | 0.813853 | 0.926174 | 0.957967 |
| Run Two | XGBoost | core + created features | 0.1 | test | 0.283463 | 148.513679 | 35.851394 | 0.082335 | 0.420670 | 0.596226 | 8.233537 | 42.066966 | 59.622642 | 1.086706 | 1.528139 | 0.558674 | 0.808314 | 0.925383 | 0.959270 |
| Run Three | RandomForestRegressor | all features | 0.2 | test | 0.210574 | 122.643537 | 24.119354 | 0.024944 | 0.239314 | 0.439023 | 2.494416 | 23.931415 | 43.902325 | 1.025220 | 1.274954 | 0.767351 | 0.904855 | 0.960015 | 0.977284 |
| Run Four | RandomForestRegressor | all features | 0.1 | test | 0.212869 | - | 23.957477 | - | - | - | - | - | - | - | - | 0.772145 | 0.906019 | 0.960015 | 0.976028 |
| Run Five | MLP | all features | 0.1 | test | 0.210124 | 180.000830 | 23.9345 | 0.029034 | 0.213476 | 0.404748 | 2.903400 | 21.347600 | 40.474800 | 1.029510 | 1.246257 | 0.794908 | 0.915654 | 0.966764 | 0.977983 |
| Run Six | MLP | all features | 0.1 (all splits) | test | 0.182711 | 107.552799 | 22.6258 | 0.028654 | 0.191106 | 0.327739 | 2.865400 | 19.110600 | 32.773900 | 1.028976 | 1.212967 | 0.816787 | 0.929560 | 0.974279 | 0.986750 |
| Run Seven | MLP | all features | 0.1 (all splits) | test | 0.184062 | 108.205227 | 22.1019 | 0.028154 | 0.193186 | 0.356171 | 2.815400 | 19.318600 | 35.617100 | 1.028566 | 1.219061 | 0.812451 | 0.924981 | 0.968872 | 0.983583 |
| Run Eight | ExtraTrees | all features | 0.1 (all splits) | test | 0.188871 | 109.896726 | 23.538942 | 0.018893 | 0.207549 | 0.373858 | 1.889280 | 20.754937 | 37.385823 | 1.019076 | 1.233467 | 0.799347 | 0.920012 | 0.966923 | 0.983827 |
| Run Nine | RandomForestRegressor | all features | 0.1 (all splits) | test | 0.188146 | 110.218555 | 23.349931 | 0.022352 | 0.209113 | 0.367556 | 2.235220 | 20.911303 | 36.755639 | 1.022575 | 1.235598 | 0.792771 | 0.921132 | 0.967605 | 0.984071 |
| Run Ten | MLP | all + engineered features | 0.1 (all splits) | test | 0.194118 | 139.257868 | 30.236400 | 0.019758 | 0.134554 | 0.314909 | 1.975800 | 13.455400 | 31.490900 | 1.019939 | 1.143091 | 0.873977 | 0.938036 | 0.967410 | 0.983194 |
| Run Eleven | MLP | all + engineered features | 0.1 (all splits) | test | 0.190830 | 130.568514 | 29.145200 | 0.032246 | 0.142126 | 0.290860 | 3.224600 | 14.212600 | 29.086000 | 1.032880 | 1.152949 | 0.850546 | 0.943005 | 0.969797 | 0.984217 |
| Run Twelve | RandomForestRegressor | all features | 0.1 (all splits) | test | 0.187844 | 108.900031 | 23.493262 | 0.020651 | 0.207398 | 0.369541 | 2.065064 | 20.739752 | 36.954097 | 1.020888 | 1.233797 | 0.796765 | 0.921327 | 0.967654 | 0.984071 |
| Run Thirteen | MLP | all features | 0.1 (all splits) | test | 0.193490 | 115.915838 | 30.038800 | 0.020046 | 0.133075 | 0.317705 | 2.004600 | 13.307500 | 31.770500 | 1.020287 | 1.142147 | 0.875633 | 0.938572 | 0.968287 | 0.983583 |
| Run Fourteen | MLP | all features | 0.1 (all splits) | test | 0.190213 | 115.071403 | 29.258100 | 0.030396 | 0.138393 | 0.286489 | 3.039600 | 13.839300 | 28.648900 | 1.031112 | 1.150090 | 0.857463 | 0.944174 | 0.970333 | 0.984704 |
| Run Fifteen | MLP | all features | 0.1 (all splits) | test | 0.204118 | 132.463814 | 28.653100 | 0.027948 | 0.141348 | 0.310609 | 2.794800 | 14.134800 | 31.060900 | 1.028575 | 1.153478 | 0.864721 | 0.939400 | 0.967654 | 0.982609 |

# Notes

*RandomForestRegressor* 

Decreasing CV to 0.05 or increasing CV to 0.4 with all features did not improve perfomance.

Using Top 10 node counts as the only node features with 0.2 CV did not improve perfomance.

Using movement_frac_x_cores and mb_per_core as additional features with 0.2 CV did not improve perfomance.

Using a ExtraTreesModel with 0.2 CV did not improve perfomance.

(Run 8) ExtraTrees with 0.1 CV dropout for all splits and all features got a 79.9% score.

(Run 12) RandomForest with log-scaled features and 0.1 CV dropout for all splits slightly beat Run 9 on rmse_log and rmse_ms, but within-10% was about the same.

*MLP*

Got upto 70% within 10 percent accuracy for the FiLM-style hardware-conditioned model tower. 

Simpler two-tower model, restricting/adding additional created features, and modifying CV did not improved perfomance.

Achieved 72.8% with 1000 epochs using FiLM-style model, 0.1 CV and all features.

Achieved 79.4% with 2000 epochs using FiLM-style model, 0.1 CV and all features.

Achieved 76.1% with 1000 epochs using FiLM-style model, no CV dropout with all features.

(Run 6) Achieved 81.6% with 2000 epochs using FiLM-style model, 0.1 CV dropout (for all splits) with all features. No big improvement after 1000 epochs.

Did not achieve any improvement by reducing all splits CV dropout to 0.05.

(Run 7) Got a 81.2% score with 0.1 CV dropout for all splits and all features for 2000 epochs with a simple one-tower MLP. No improvement after ~1000 epochs.

(Run 10) Gated FiLM MLP with engineered model/hardware features got the best within-10% score so far at 87.4%, but rmse_ms and rmse_percent were worse than Run 6.

(Run 11) Same gated FiLM setup as Run 10, but selected by validation rmse_log; test rmse_log and rmse_percent improved over Run 10, while within-10% dropped to 85.1%.

(Run 13) Same gated FiLM setup as Run 10 without the extra engineered features; within-10% improved slightly to 87.6% and rmse_ms improved, but rmse_percent remained worse than Run 6.

Replicated Run 13 without log scaling of features; perfomance was much worse. 

(Run 14) Same base-feature gated FiLM setup as Run 13 with log scaling restored and checkpoint selection prioritized for within-25%; within-25%, rmse_log, rmse_ms, and rmse_percent improved over Run 13, while within-10% dropped.

(Run 15) Composite checkpoint selection improved rmse_percent over Runs 13 and 14, but hurt rmse_log, rmse_ms, within-25%, and within-2x; this does not look like a better overall direction.
