from eval.metrics import compute_metrics, compute_metrics_by_transport


def test_metrics_scale_to_fifty_result_matrix():
    results = [
        {
            "runtime": f"runtime_{runtime}",
            "transport": f"transport_{transport}",
            "task_id": f"t{task}_synthetic",
            "success": True,
            "latency_ms": float(runtime + transport + task),
            "approvals": [],
            "metadata": {"tool_durations_ms": {"fs_read": [1.0, 2.0]}},
            "trace_completeness": 1.0,
        }
        for runtime in range(5)
        for transport in range(5)
        for task in range(2)
    ]

    bundle = compute_metrics(results)
    by_transport = compute_metrics_by_transport(results)

    assert len(results) == 50
    assert bundle.trace_completeness == 1.0
    assert bundle.latency_p95 > 0
    assert len(by_transport) == 5
