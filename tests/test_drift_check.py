from monitoring.drift_check import _extract_drift_summary


def _result(drifted, total, threshold=0.5):
    # shape of Evidently 0.4.x DatasetDriftMetric: `drift_share` is the THRESHOLD, not the measurement
    return {"metrics": [{"result": {
        "dataset_drift": drifted / total >= threshold, "drift_share": threshold,
        "share_of_drifted_columns": drifted / total,
        "number_of_columns": total, "number_of_drifted_columns": drifted, "drift_by_columns": {}}}]}


def test_share_is_the_measured_share_not_the_threshold():
    assert _extract_drift_summary(_result(11, 68))["drift_share"] == 11 / 68


def test_share_is_derived_from_counts_even_if_the_share_key_is_missing():
    r = _result(34, 68)
    del r["metrics"][0]["result"]["share_of_drifted_columns"]
    assert _extract_drift_summary(r)["drift_share"] == 0.5
