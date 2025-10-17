from eval.metrics import (
    MetricsBundle,
    bundle_to_dict,
    compare_to_baseline,
    compute_latency_alerts,
    compute_metrics_by_transport,
    compute_portability,
    compute_success_alerts,
    compute_stdio_wait_alerts,
)


def make_row(transport: str, latency: float, success: bool = True, task_id: str = "t1_repo_triage") -> dict:
    return {
        "runtime": f"runtime_{transport}",
        "task_id": task_id,
        "policy_mode": "never",
        "success": success,
        "latency_ms": latency,
        "tokens_prompt": 1,
        "tokens_completion": 1,
        "tool_calls": 1,
        "approvals": [],
        "failure_category": None,
        "trace_completeness": 1.0,
        "transport": transport,
        "metadata": {},
    }


def test_compute_metrics_by_transport_splits_results():
    rows = [
        make_row("embedded", 10.0),
        make_row("embedded", 12.0),
        make_row("http", 30.0),
        make_row("http", 32.0),
    ]
    grouped = compute_metrics_by_transport(rows)
    assert set(grouped.keys()) == {"embedded", "http"}
    assert isinstance(grouped["embedded"], MetricsBundle)
    assert grouped["embedded"].latency_p50 == 11.0
    assert grouped["http"].latency_p50 == 31.0


def test_bundle_to_dict_contains_tool_data():
    bundle = MetricsBundle(
        portability_success_rate=1.0,
        trace_completeness=1.0,
        latency_p50=10.0,
        latency_p95=12.0,
        approvals=2,
        token_total=4,
        failure_histogram={"runtime": 1},
        tool_latency_p50={"fs_read": 5.0},
        tool_latency_p95={"fs_read": 6.0},
        tool_latency_count={"fs_read": 3},
    )
    data = bundle_to_dict(bundle)
    assert data["tool_latency_p50"]["fs_read"] == 5.0
    assert data["failure_histogram"]["runtime"] == 1


def test_compute_latency_alerts_flags_exceeding_threshold():
    metrics_map = {
        "embedded": MetricsBundle(
            portability_success_rate=1.0,
            trace_completeness=1.0,
            latency_p50=10.0,
            latency_p95=20.0,
            approvals=0,
            token_total=0,
            failure_histogram={},
            tool_latency_p50={},
            tool_latency_p95={},
            tool_latency_count={},
        ),
        "http": MetricsBundle(
            portability_success_rate=1.0,
            trace_completeness=1.0,
            latency_p50=10.0,
            latency_p95=120.0,
            approvals=0,
            token_total=0,
            failure_histogram={},
            tool_latency_p50={},
            tool_latency_p95={},
            tool_latency_count={},
        ),
    }
    alerts = compute_latency_alerts(metrics_map, threshold_ms=100.0)
    assert alerts == [
        {
            "transport": "http",
            "latency_p95": 120.0,
            "threshold_ms": 100.0,
        }
    ]


def test_compute_success_alerts_flags_underperforming_transports():
    rows = [
        make_row("embedded", 10.0, success=True),
        make_row("embedded", 10.0, success=False),
        make_row("http", 10.0, success=False),
        make_row("http", 10.0, success=False),
    ]
    alerts = compute_success_alerts(rows, threshold=0.6)
    assert alerts == [
        {"transport": "embedded", "success_rate": 0.5, "threshold": 0.6},
        {"transport": "http", "success_rate": 0.0, "threshold": 0.6},
    ]


def test_compare_to_baseline_flags_regressions():
    current = {
        "embedded": MetricsBundle(
            portability_success_rate=1.0,
            trace_completeness=1.0,
            latency_p50=10.0,
            latency_p95=120.0,
            approvals=0,
            token_total=0,
            failure_histogram={},
            tool_latency_p50={},
            tool_latency_p95={},
            tool_latency_count={},
        )
    }
    baseline = {"embedded": {"latency_p95": 100.0, "portability_success_rate": 1.0}}
    alerts = compare_to_baseline(current, baseline, max_latency_delta=10.0, max_success_delta=0.05)
    assert alerts == [
        {"transport": "embedded", "latency_p95_delta": 20.0, "max_latency_delta": 10.0}
    ]


def test_compute_stdio_wait_alerts():
    transports = {
        "stdio": {
            "stdio_pool": {
                "wait_ms": {"p95": 120.0}
            }
        }
    }
    baseline = {
        "stdio": {
            "stdio_pool": {
                "wait_ms": {"p95": 110.0}
            }
        }
    }
    alerts = compute_stdio_wait_alerts(transports, threshold_ms=100.0, baseline=baseline)
    assert alerts == [
        {
            "transport": "stdio",
            "wait_p95": 120.0,
            "threshold_ms": 100.0,
            "delta_wait_p95": 10.0,
        }
    ]


def test_compute_stdio_wait_alerts_triggers_on_regression_without_threshold_breach():
    transports = {
        "stdio": {
            "stdio_pool": {
                "wait_ms": {"p95": 80.0}
            }
        }
    }
    baseline = {
        "stdio": {
            "stdio_pool": {
                "wait_ms": {"p95": 60.0}
            }
        }
    }
    alerts = compute_stdio_wait_alerts(transports, threshold_ms=100.0, baseline=baseline)
    assert alerts == [
        {
            "transport": "stdio",
            "wait_p95": 80.0,
            "threshold_ms": 100.0,
            "delta_wait_p95": 20.0,
        }
    ]


def test_compute_portability_includes_write_task():
    results = [
        make_row("embedded", 10.0, success=True, task_id="t1_repo_triage"),
        make_row("embedded", 12.0, success=True, task_id="t2_calendar_merge"),
        make_row("embedded", 14.0, success=True, task_id="t3_http_etl"),
        make_row("embedded", 16.0, success=False, task_id="t4_code_patch"),
    ]
    portability = compute_portability(results)
    assert portability == 0.75
