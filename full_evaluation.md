# Full evaluation of every tool (measured in CI on real libraries)

## Step status

| job | step | passed | time |
|---|---|---|---|
| airflow | airflow_image | yes | 68 s |
| airflow | airflow_db_migrate | yes | 9 s |
| airflow | airflow_idle | yes | 18 s |
| airflow | airflow_drift | yes | 42 s |
| core | dvc | yes | 62 s |
| core | mlflow | yes | 2 s |
| core | mlflow_replay | yes | 3 s |
| core | events | yes | 3 s |
| core | curve | yes | 5 s |
| core | spread | yes | 74 s |
| core | onnx_convert | yes | 3 s |
| core | onnx | yes | 4 s |
| core | api | yes | 41 s |
| core | load | yes | 6 s |
| core | load_baseline | yes | 55 s |
| core | load_onnx | yes | 6 s |
| core | batch_profile_memory | yes | 4 s |
| core | batch_profile_redis | yes | 8 s |
| core | batch_profile_onnx | yes | 8 s |
| core | stages | yes | 14 s |
| core | compact | yes | 39 s |
| core | drift_demo | yes | 10 s |
| core | docker | yes | 162 s |
| k8s | k8s_hpa | yes | 980 s |
| triton | triton_up | yes | 135 s |
| triton | triton | yes | 5 s |

## Airflow run: airflow_idle

| measure | value |
|---|---|
| tasks succeeded | 3 |
| tasks failed | 0 |
| tasks skipped | 0 |
| tasks seen | decide_whether_to_retrain, run_drift_check, skip_retrain |

## Airflow run: airflow_drift

| measure | value |
|---|---|
| tasks succeeded | 6 |
| tasks failed | 0 |
| tasks skipped | 0 |
| tasks seen | decide_whether_to_retrain, engineer_features, generate_data, promote_model, run_drift_check, train_model |

## Model on the held-out window (real XGBoost)

| metric | value |
|---|---|
| interval recall | 0.9504 |
| interval precision | 0.8038 |
| interval F1 | 0.871 |
| PR-AUC | 0.967 |
| ROC-AUC | 0.9966 |
| incident detection rate | 1 |
| median detection delay (intervals) | 0 |
| false alerts / cell / day | 0.3084 |
| threshold | 0.2277 |

## Precision/recall trade-off (threshold tuned on validation, applied to test)

| recall target | threshold | val recall / precision | test recall / precision | test F1 | incidents detected | false alerts/cell/day |
|---|---|---|---|---|---|---|
| 0.9 | 0.7623 | 0.9009 / 0.9738 | 0.9184 / 0.9729 | 0.9449 | 1 | 0.037 |
| 0.93 | 0.4883 | 0.9317 / 0.9256 | 0.9296 / 0.9341 | 0.9318 | 1 | 0.089 |
| 0.935 | 0.3995 | 0.9361 / 0.8929 | 0.936 / 0.907 | 0.9213 | 1 | 0.133 |
| 0.94 | 0.3432 | 0.9405 / 0.8472 | 0.944 / 0.8806 | 0.9112 | 1 | 0.176 |
| 0.945 | 0.2277 | 0.9471 / 0.7414 | 0.9504 / 0.8038 | 0.871 | 1 | 0.308 |
| 0.95 | 0.1086 | 0.9515 / 0.5752 | 0.9616 / 0.6215 | 0.755 | 1 | 0.777 |
| 0.955 | 0.0849 | 0.9559 / 0.5161 | 0.9632 / 0.561 | 0.7091 | 1 | 1.003 |
| 0.96 | 0.06414 | 0.9604 / 0.4523 | 0.9664 / 0.4959 | 0.6555 | 1 | 1.284 |
| 0.965 | 0.04746 | 0.967 / 0.393 | 0.9728 / 0.4276 | 0.594 | 1 | 1.675 |
| 0.98 | 0.02886 | 0.9802 / 0.2969 | 0.9792 / 0.3346 | 0.4988 | 1 | 2.39 |

Best precision any threshold could reach on test at a given recall: recall >= 0.8: 1.0, recall >= 0.9: 0.9843, recall >= 0.93: 0.9312, recall >= 0.95: 0.8227, recall >= 0.97: 0.4776, recall >= 0.98: 0.328, recall >= 0.99: 0.2427

## Real XGBoost across 10 independent datasets (recall target 0.945)

| metric | min | median | max |
|---|---|---|---|
| event_detection_rate | 1 | 1 | 1 |
| f1 | 0.864 | 0.907 | 0.939 |
| false_alerts_per_cell_day | 0.071 | 0.19 | 0.308 |
| pr_auc | 0.961 | 0.966 | 0.972 |
| precision | 0.801 | 0.874 | 0.949 |
| recall | 0.923 | 0.942 | 0.952 |

| seed | recall | precision | F1 |
|---|---|---|---|
| 1 | 0.9485 | 0.8425 | 0.8924 |
| 2 | 0.9387 | 0.8743 | 0.9053 |
| 3 | 0.9311 | 0.9114 | 0.9211 |
| 4 | 0.9289 | 0.9486 | 0.9386 |
| 42 | 0.9504 | 0.8038 | 0.871 |
| 5 | 0.9514 | 0.8879 | 0.9185 |
| 6 | 0.9363 | 0.8015 | 0.8637 |
| 7 | 0.9232 | 0.9302 | 0.9267 |
| 8 | 0.9515 | 0.8538 | 0.9 |
| 9 | 0.9455 | 0.8735 | 0.9081 |

## FastAPI

| check | passed |
|---|---|
| /model/info threshold == metadata threshold | yes |
| API scores every held-out row | yes |
| API scores match offline model (max diff < 1e-3) | yes |
| GET /health -> 200 | yes |
| GET /metrics -> 200 (Prometheus text) | yes |
| after warm-up raw readings -> 200 with live features | yes |
| auth: /health stays open | yes |
| auth: /metrics stays open | yes |
| auth: first key -> 200 | yes |
| auth: no key -> 401 | yes |
| auth: second (rotation) key -> 200 | yes |
| auth: wrong key -> 401 | yes |
| first raw reading for a new cell -> 425 (no guessing) | yes |
| invalid reading (negative latency) -> 422 | yes |

| measure | value |
|---|---|
| held-out rows scored through the API | 42021 |
| max |API score - offline score| | 5e-05 |
| decision agreement | 1 |
| API recall / precision / F1 | 0.9504 / 0.8038 / 0.871 |
| batch throughput (rows/s, 500 per call) | 1282 |
| 501-snapshot batch status | 400 |

## API under load (1 process, Redis-backed)

| measure | value |
|---|---|
| idle_rss_mib | 277 |
| peak_rss_mib | 280 |
| req_per_s | 273 |
| cpu_ms_per_request | 4.6 |
| p50_ms | 14 |
| p95_ms | 19 |
| redis_mib | 5.1 |

## Optimisation: whole API under the same load

| API path | idle_rss_mib | peak_rss_mib | req_per_s | cpu_ms_per_request | p50_ms | p95_ms |
|---|---|---|---|---|---|---|
| original | 278 | 284 | 23 | 50.3 | 168 | 228 |
| optimised | 277 | 280 | 273 | 4.6 | 14 | 19 |
| optimised + ONNX scorer | 296 | 298 | 313 | 3.7 | 12 | 17 |

## Optimisation: cost of each stage of one request (ms)

| stage | ms |
|---|---|
| features_numpy_ms | 0.6494 |
| features_pandas_ms | 25.06 |
| onnx_predict_ms | 0.014 |
| scaler_numpy_ms | 0.0027 |
| scaler_sklearn_ms | 0.1038 |
| xgboost_predict_ms | 0.4237 |

## Smaller models (same data, threshold tuned on validation to the same recall target)

| model | trees used | test recall | test precision | test F1 | model.json KiB | predict one row ms |
|---|---|---|---|---|---|---|
| depth3_trees300 | 285 | 0.96 | 0.722 | 0.8242 | 365 | 0.403 |
| depth4_trees150 | 143 | 0.952 | 0.7757 | 0.8549 | 308 | 0.401 |
| depth4_trees300 | 189 | 0.9536 | 0.7873 | 0.8625 | 447 | 0.403 |
| depth5_trees300 | 189 | 0.9504 | 0.7734 | 0.8528 | 734 | 0.413 |
| depth6_trees100 | 99 | 0.952 | 0.6357 | 0.7623 | 544 | 0.413 |
| depth6_trees150 | 143 | 0.9552 | 0.7416 | 0.835 | 796 | 0.406 |
| depth6_trees300 | 189 | 0.9504 | 0.8038 | 0.871 | 1127 | 0.415 |

## ONNX Runtime vs XGBoost

| measure | value |
|---|---|
| max |prob diff| | 1.304e-07 |
| decision agreement | 1 |
| model.json / model.onnx (KiB) | 1127.3 / 481.4 |
| single row median ms (xgboost / onnx) | 0.415 / 0.02 |
| 500-row batch median ms (xgboost / onnx) | 1.215 / 0.978 |
| onnx outputs | label, probabilities |

## MLflow

| item | value |
|---|---|
| experiments | {"Default": 0, "telecom-anomaly-detection": 3} |
| latest_run | {"artifacts": ["feature_importance.csv", "metadata.json", "model", "scaler.pkl"], "id": "7db06c52e0654216b5f30ab6893a7f6b", "metrics": {"best_iteration": 188.0, "f1": 0.871, "false_negatives": 31.0, "false_positives": 145.0, "precision": 0.8038, "recall": 0.9504, "test_pr_auc": 0.967, "threshold": 0.2277, "true_negatives": 41251.0, "true_positives": 594.0, "val_f1": 0.8317, "val_pr_auc": 0.9585, "val_precision": 0.7414, "val_recall": 0.9471}, "params_logged": 19, "status": "FINISHED"} |
| production_versions_after_gate | [3] |
| promotion_gate | {"promoted": true, "recall": 0.9504, "version": 3} |
| registered_versions | [1, 2, 3] |
| registry_roundtrip_max_prob_diff | 0 |
| tracking_uri | file:///home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/mlruns |
| training_runs | 3 |

## DVC

| run | exit | seconds | stages |
|---|---|---|---|
| first_repro | 0 | 28 | ran:generate_data ran:features ran:train |
| second_repro_no_changes | 0 |  | skipped:generate_data skipped:features skipped:train |
| after_changing_validation_recall_target_0.945_to_0.94 | 0 | 15 | skipped:generate_data skipped:features ran:train |
| restore_params_repro | 0 |  | skipped:generate_data skipped:features ran:train |

## Docker image

| measure | value |
|---|---|
| build_seconds_cold | 26 |
| image_mib | 487.2 |
| drift_image_mib | 895.4 |
| full_requirements_image_mib | 3011 |
| layers | 12 |
| healthcheck_status | healthy |
| seconds_to_healthy | 6 |
| runs_as_uid | 1000 |
| health_http | 200 |
| container_memory_idle | 121.1MiB / 15.61GiB |
| largest_site_packages_mib | ['109 /usr/local/lib/python3.12/site-packages/scipy', '75 /usr/local/lib/python3.12/site-packages/pandas', '57 /usr/local/lib/python3.12/site-packages/sklearn', '43 /usr/local/lib/python3.12/site-packages/numpy', '30 /usr/local/lib/python3.12/site-packages/scipy.libs', '27 /usr/local/lib/python3.12/site-packages/numpy.libs', '15 /usr/local/lib/python3.12/site-packages/xgboost', '14 /usr/local/lib/python3.12/site-packages/pip', '6 /usr/local/lib/python3.12/site-packages/pydantic_core', '5 /usr/local/lib/python3.12/site-packages/pydantic', '4 /usr/local/lib/python3.12/site-packages/yaml', '3 /usr/local/lib/python3.12/site-packages/tzdata', '3 /usr/local/lib/python3.12/site-packages/redis', '3 /usr/local/lib/python3.12/site-packages/pytz'] |

## Evidently drift detection

| scenario | dataset drift | features flagged | share |
|---|---|---|---|
| congestion_incident | yes | 55/68 | 0.809 |
| healthy_holdout | NO | 0/68 | 0 |
| latency_x2_only | NO | 11/68 | 0.162 |

## Batch endpoint profile: batch_profile_memory

```
/opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa
history backend: memory
batch of 500: 647 ms total, 1.29 ms per reading (no profiler)
=> 773 readings/s in one process, 9 s for 7,000 cells

         720051 function calls (719991 primitive calls) in 0.924 seconds

   Ordered by: cumulative time
   List reduced from 815 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.000    0.000    0.920    0.920 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/scripts/profile_batch.py:46(batch)
      2/1    0.000    0.000    0.917    0.917 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:1123(post)
        1    0.000    0.000    0.915    0.915 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/starlette/testclient.py:493(request)
        1    0.000    0.000    0.915    0.915 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:771(request)
        1    0.000    0.000    0.915    0.915 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:879(send)
        1    0.000    0.000    0.914    0.914 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:930(_send_handling_auth)
        1    0.000    0.000    0.914    0.914 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:964(_send_handling_redirects)
        1    0.000    0.000    0.914    0.914 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:1001(_send_single_request)
    57/54    0.001    0.000    0.913    0.017 {method 'run' of '_contextvars.Context' objects}
        1    0.000    0.000    0.912    0.912 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/starlette/testclient.py:234(handle_request)
        7    0.000    0.000    0.912    0.130 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/anyio/from_thread.py:332(call)
        1    0.002    0.002    0.902    0.902 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/main.py:313(predict_batch)
      500    0.010    0.000    0.835    0.002 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/main.py:238(run_inference)
      500    0.004    0.000    0.541    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/main.py:170(build_feature_vector)
      500    0.002    0.000    0.534    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:270(add_and_compute)
      500    0.052    0.000    0.532    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:47(fast_latest_features)
      500    0.002    0.000    0.263    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/scorer.py:53(probability)
      500    0.002    0.000    0.257    0.001 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/xgboost/sklearn.py:1660(predict_proba)
      500    0.003    0.000    0.242    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/xgboost/sklearn.py:1205(predict)
      500    0.113    0.000    0.191    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/xgboost/core.py:2397(inplace_predict)
```

## Batch endpoint profile: batch_profile_redis

```
/opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa
history backend: redis
batch of 500: 744 ms total, 1.49 ms per reading (no profiler)
=> 672 readings/s in one process, 10 s for 7,000 cells

         1236598 function calls (1222039 primitive calls) in 1.164 seconds

   Ordered by: cumulative time
   List reduced from 868 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.000    0.000    1.160    1.160 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/scripts/profile_batch.py:46(batch)
      2/1    0.000    0.000    1.157    1.157 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:1123(post)
        1    0.000    0.000    1.155    1.155 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/starlette/testclient.py:493(request)
        1    0.000    0.000    1.155    1.155 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:771(request)
        1    0.000    0.000    1.155    1.155 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:879(send)
        1    0.000    0.000    1.154    1.154 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:930(_send_handling_auth)
        1    0.000    0.000    1.154    1.154 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:964(_send_handling_redirects)
        1    0.000    0.000    1.154    1.154 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/httpx/_client.py:1001(_send_single_request)
    57/54    0.004    0.000    1.152    0.021 {method 'run' of '_contextvars.Context' objects}
        7    0.000    0.000    1.152    0.165 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/anyio/from_thread.py:332(call)
        1    0.000    0.000    1.150    1.150 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/starlette/testclient.py:234(handle_request)
        1    0.003    0.003    1.139    1.139 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/main.py:313(predict_batch)
      500    0.009    0.000    0.825    0.002 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/main.py:238(run_inference)
      500    0.004    0.000    0.535    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/main.py:170(build_feature_vector)
      500    0.002    0.000    0.528    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:270(add_and_compute)
      500    0.050    0.000    0.526    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:47(fast_latest_features)
      500    0.002    0.000    0.259    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/scorer.py:53(probability)
        1    0.000    0.000    0.255    0.255 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:265(append_many)
        1    0.005    0.005    0.255    0.255 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:212(append_many)
      500    0.002    0.000    0.253    0.001 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/xgboost/sklearn.py:1660(predict_proba)
```

## Batch endpoint profile: batch_profile_onnx

```
/opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa
history backend: redis
batch of 500: 535 ms total, 1.07 ms per reading (no profiler)
=> 934 readings/s in one process, 7 s for 7,000 cells

         1150084 function calls (1135503 primitive calls) in 0.907 seconds

   Ordered by: cumulative time
   List reduced from 801 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       14    0.001    0.000    3.799    0.271 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/main.py:238(run_inference)
      500    0.004    0.000    0.552    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/scorer.py:53(probability)
      500    0.004    0.000    0.521    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/main.py:170(build_feature_vector)
      500    0.001    0.000    0.514    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:270(add_and_compute)
      500    0.049    0.000    0.513    0.001 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:47(fast_latest_features)
        1    0.000    0.000    0.253    0.253 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:265(append_many)
        1    0.005    0.005    0.253    0.253 /home/runner/work/telecom-network-anomaly-detection/telecom-network-anomaly-detection/api/live_features.py:212(append_many)
     5000    0.006    0.000    0.163    0.000 {method 'std' of 'numpy.ndarray' objects}
        3    0.000    0.000    0.160    0.053 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/redis/retry.py:47(call_with_retry)
        2    0.000    0.000    0.160    0.080 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/redis/client.py:1570(execute)
        2    0.000    0.000    0.160    0.080 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/redis/client.py:1591(<lambda>)
        2    0.006    0.003    0.160    0.080 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/redis/client.py:1423(_execute_transaction)
     5000    0.014    0.000    0.158    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/numpy/core/_methods.py:204(_std)
     5000    0.074    0.000    0.141    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/numpy/core/_methods.py:135(_var)
      500    0.005    0.000    0.118    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/pandas/core/tools/datetimes.py:678(to_datetime)
      500    0.004    0.000    0.100    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/pandas/core/tools/datetimes.py:316(_convert_listlike_datetimes)
    11002    0.012    0.000    0.099    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/json/__init__.py:299(loads)
      500    0.012    0.000    0.088    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/pandas/core/tools/datetimes.py:458(_array_strptime_with_fallback)
     3505    0.002    0.000    0.075    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/redis/client.py:1522(parse_response)
     3506    0.003    0.000    0.074    0.000 /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/redis/client.py:628(parse_response)
```

## Kubernetes: HPA scale-out / scale-in (kind, metrics-server)

| measure | value |
|---|---|
| HPA (min / max / CPU target) | 2 / 6 / 70% |
| load generator (24 threads, 240 s) | 631.6 req/s, p50 35 ms, p95 73 ms, errors 0 |
| seconds from load start to first scale-out decision | 31 |
| max desired / max ready replicas | 6 / 5 |
| peak average CPU utilisation (of the 500m request) | 188% |
| max Pending API pods (insufficient CPU on the runner) | 1 |
| peak memory of one API pod (MiB) | 135 |
| seconds from load end to first scale-in | 220 |
| seconds from load end back at min replicas | 388 |

## Kubernetes: per-component footprint (peak CPU millicores / memory MiB)

| component | idle | under load | after load |
|---|---|---|---|
| anomaly-api | 2m / 122 Mi | 975m / 135 Mi | 560m / 135 Mi |
| grafana | 1m / 62 Mi | 3m / 62 Mi | 3m / 53 Mi |
| loadgen | Nonem / None Mi | 878m / 31 Mi | Nonem / None Mi |
| prometheus | 1m / 24 Mi | 6m / 36 Mi | 8m / 39 Mi |
| pushgateway | 1m / 8 Mi | 1m / 11 Mi | 1m / 13 Mi |
| redis | 7m / 3 Mi | 117m / 65 Mi | 116m / 64 Mi |

| measure | value |
|---|---|
| Prometheus head series | 2300 |
| samples / s | 153.3 |
| TSDB on disk (KiB) after the run | 312 |
| estimated disk for 15 d retention (MiB) | 284 |
| drift job result | drift job: succeeded=1 failed=0 after 17 s |
| drift job peak CPU / memory | Nonem / None Mi |

## Triton (CPU) vs XGBoost

| measure | value |
|---|---|
| max |prob diff| | 1.136e-07 |
| decision agreement | 1 |
| output names | probabilities |
| HTTP single row median / p95 ms | 6.35 / 6.64 |
| HTTP 64-row batch median ms | 9 |
