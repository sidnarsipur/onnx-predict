# Run results

| Run | Model | Features | CV | split | rmse_log | rmse_ms | rmse_percent | median_relative_error | p90_relative_error | p95_relative_error | median_percent_error | p90_percent_error | p95_percent_error | median_ratio_error | p90_ratio_error | within_10pct | within_25pct | within_50pct | within_2x |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run One | XGBoost | core features | 0.1 | test | 0.282506 | 132.612615 | 36.069926 | 0.078311 | 0.414712 | 0.590687 | 7.831146 | 41.471205 | 59.068737 | 1.081896 | 1.523052 | 0.572499 | 0.813853 | 0.926174 | 0.957967 |
| Run Two | XGBoost | core + created features | 0.1 | test | 0.283463 | 148.513679 | 35.851394 | 0.082335 | 0.420670 | 0.596226 | 8.233537 | 42.066966 | 59.622642 | 1.086706 | 1.528139 | 0.558674 | 0.808314 | 0.925383 | 0.959270 |
| Run Three | RandomForestRegressor | all features | 0.2 | test | 0.210574 | 122.643537 | 24.119354 | 0.024944 | 0.239314 | 0.439023 | 2.494416 | 23.931415 | 43.902325 | 1.025220 | 1.274954 | 0.767351 | 0.904855 | 0.960015 | 0.977284 |
| Run Four | RandomForestRegressor | all features | 0.1 | test | 0.212869 | - | 23.957477 | - | - | - | - | - | - | - | - | 0.772145 | 0.906019 | 0.960015 | 0.976028 |

# Notes

*RandomForestRegressor* 

Decreasing CV to 0.05 or increasing CV to 0.4 with all features did not improve perfomance.

Using Top 10 node counts as the only node features with 0.2 CV did not improve perfomance.

Using movement_frac_x_cores and mb_per_core as additional features with 0.2 CV did not improve perfomance.

Using a ExtraTreesModel with 0.2 CV did not improve perfomance.