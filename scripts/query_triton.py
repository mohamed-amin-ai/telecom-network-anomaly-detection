"""
query_triton.py
-----------------
Sends a real inference request directly to Triton (KServe v2 HTTP
protocol) using a real held-out row from features.csv, and — if the
FastAPI service is also reachable — compares Triton's answer against
/predict for the exact same row. This is the actual point of the ONNX +
Triton path: proving the two serving routes agree, not just that Triton
starts.

Usage:
  python scripts/query_triton.py
  python scripts/query_triton.py --triton-url http://localhost:8003 --api-url http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))
from data.features import get_feature_columns
from models.common import load_params, load_data, time_split


def build_infer_request(feature_vector: list, input_name: str = "input") -> dict:
    """Build a KServe v2 HTTP inference request body. Pure function, no
    network — testable without a live Triton server (see
    tests/test_query_triton.py)."""
    return {
        "inputs": [
            {
                "name": input_name,
                "shape": [1, len(feature_vector)],
                "datatype": "FP32",
                "data": [float(x) for x in feature_vector],
            }
        ]
    }


def parse_infer_response(response_json: dict) -> dict:
    """Pull {output_name: [values]} out of a KServe v2 response body.
    Pure function — testable without a live server."""
    return {out["name"]: out["data"] for out in response_json.get("outputs", [])}


def get_one_real_row():
    params = load_params()
    df = load_data()
    feature_cols = get_feature_columns(df)
    _, test_df = time_split(df, params["split"]["train_ratio"])
    row = test_df[feature_cols].iloc[0]
    return row.values.tolist(), feature_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-url", default="http://localhost:8003")
    parser.add_argument("--model-name", default="telecom_anomaly_xgb")
    parser.add_argument("--api-url", default="http://localhost:8000",
                         help="If reachable, used to cross-check Triton's answer against /predict.")
    args = parser.parse_args()

    feature_vector, feature_cols = get_one_real_row()
    print(f"Using 1 real held-out row across {len(feature_cols)} features.")

    infer_url = f"{args.triton_url}/v2/models/{args.model_name}/infer"
    request_body = build_infer_request(feature_vector)

    try:
        resp = requests.post(infer_url, json=request_body, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Could not reach Triton at {infer_url} — is docker-compose.triton.yml running? ({e})")
        return

    outputs = parse_infer_response(resp.json())
    print(f"Triton outputs: {outputs}")

    probs = outputs.get("probabilities")
    if probs is None:
        print("No 'probabilities' output found — config.pbtxt's output name likely "
              "doesn't match what conversion actually produced. Check "
              "`python models/convert_to_onnx.py`'s printed output names.")
        return
    triton_score = probs[1]  # [P(normal), P(anomaly)]
    print(f"Triton anomaly_score = {triton_score:.4f}")

    # Cross-check against the FastAPI service, same row, precomputed
    # features so there's no cold-start/live-history complication.
    try:
        features_dict = dict(zip(feature_cols, feature_vector))
        api_resp = requests.post(
            f"{args.api_url}/predict",
            json={
                "cell_id": "TRITON_PARITY_CHECK",
                "latency_ms": 0, "throughput_mbps": 0, "packet_loss_pct": 0,
                "sinr_db": 0, "connected_users": 0,  # ignored when `features` is set
                "features": features_dict,
            },
            timeout=10,
        )
        api_resp.raise_for_status()
        api_score = api_resp.json()["anomaly_score"]
        print(f"API (/predict) anomaly_score = {api_score:.4f}")
        diff = abs(triton_score - api_score)
        print(f"Difference: {diff:.6f} "
              f"({'looks consistent' if diff < 0.01 else 'LARGER than expected — investigate'})")
    except requests.exceptions.RequestException as e:
        print(f"(API at {args.api_url} not reachable, skipping cross-check: {e})")


if __name__ == "__main__":
    main()
