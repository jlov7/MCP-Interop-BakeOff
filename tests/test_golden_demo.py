from usb_agents.golden import create_golden_demo


def test_golden_demo_writes_release_summary(tmp_path):
    run = create_golden_demo(
        transports=["embedded"],
        run_root=tmp_path / "golden",
        timeout_seconds=30,
    )

    assert run.status == "passed"
    assert any(artifact.path == "release-demo.md" for artifact in run.artifacts)
    assert (tmp_path / "golden" / "release-demo.md").exists()
