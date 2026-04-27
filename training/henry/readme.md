Model Output

Train on log_latency instead of latency 

z = log(latency_ms)

Model Architecture

LightGBM or XGBoost
MLP 

Comparison across architectures

RMSE on log latency

error_i = log(predicted_latency_ms_i) - log(true_latency_ms_i)
RMSE_log = sqrt(mean(error_i²))
