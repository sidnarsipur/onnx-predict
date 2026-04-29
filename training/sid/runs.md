# Run results

| Run | Model | Features | CV | split | rmse_log | rmse_ms | rmse_percent | median_relative_error | p90_relative_error | p95_relative_error | median_percent_error | p90_percent_error | p95_percent_error | median_ratio_error | p90_ratio_error | within_10pct | within_25pct | within_50pct | within_2x |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run One | XGBoost | core features | 0.1 | test | 0.282506 | 132.612615 | 36.069926 | 0.078311 | 0.414712 | 0.590687 | 7.831146 | 41.471205 | 59.068737 | 1.081896 | 1.523052 | 0.572499 | 0.813853 | 0.926174 | 0.957967 |
| Run Two | XGBoost | core + created features | 0.1 | test | 0.283463 | 148.513679 | 35.851394 | 0.082335 | 0.420670 | 0.596226 | 8.233537 | 42.066966 | 59.622642 | 1.086706 | 1.528139 | 0.558674 | 0.808314 | 0.925383 | 0.959270 |
| Run Three | RandomForestRegressor | all features | 0.2 | test | 0.210574 | 122.643537 | 24.119354 | 0.024944 | 0.239314 | 0.439023 | 2.494416 | 23.931415 | 43.902325 | 1.025220 | 1.274954 | 0.767351 | 0.904855 | 0.960015 | 0.977284 |
| Run Four | RandomForestRegressor | all features | 0.1 | test | 0.212869 | - | 23.957477 | - | - | - | - | - | - | - | - | 0.772145 | 0.906019 | 0.960015 | 0.976028 |
| Run Five | MLP | all features | 0.1 | test | 0.210124 | 180.000830 | 23.9345 | 0.029034 | 0.213476 | 0.404748 | 2.903400 | 21.347600 | 40.474800 | 1.029510 | 1.246257 | 0.794908 | 0.915654 | 0.966764 | 0.977983 |
| Run Six | MLP | all features | 0.1 (all splits) | test | 0.182711 | 107.552799 | 22.6258 | 0.028654 | 0.191106 | 0.327739 | 2.865400 | 19.110600 | 32.773900 | 1.028976 | 1.212967 | 0.816787 | 0.929560 | 0.974279 | 0.986750 |

# Notes

*RandomForestRegressor* 

Decreasing CV to 0.05 or increasing CV to 0.4 with all features did not improve perfomance.

Using Top 10 node counts as the only node features with 0.2 CV did not improve perfomance.

Using movement_frac_x_cores and mb_per_core as additional features with 0.2 CV did not improve perfomance.

Using a ExtraTreesModel with 0.2 CV did not improve perfomance.

*MLP*

Got upto 70% within 10 percent accuracy for the FiLM-style hardware-conditioned model tower. 

Simpler two-tower model, restricting/adding features, and modifying CV did not improved perfomance.

Achieved 72.8% with 1000 epochs using FiLM-style model, 0.1 CV and all features.

Achieved 79.4% with 2000 epochs using FiLM-style model, 0.1 CV and all features.

Achieved 76.1% with 1000 epochs using FiLM-style model, no CV dropout with all features.

(Run 6) Achieved 81.6% with 2000 epochs using FiLM-style model, 0.1 CV dropout (for all splits) with all features. No big improvement after 1000 epochs.

Did not achieve any improvement by reducing all splits CV dropout to 0.05.

(Run 7) Did not achieve any improvement by using a simple one-tower MLP model.