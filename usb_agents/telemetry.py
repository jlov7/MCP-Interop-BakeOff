"""Telemetry helpers for configuring OpenTelemetry with GenAI semantics."""

from __future__ import annotations

import os
from typing import Dict, Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExportResult,
    SpanExporter,
)

try:  # pragma: no cover - optional dependency surface
    from opentelemetry.sdk.trace.export import InMemorySpanExporter
except ImportError:  # pragma: no cover - fallback
    class InMemorySpanExporter(SpanExporter):
        def __init__(self) -> None:
            self._spans = []

        def export(self, spans):
            self._spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            self._spans.clear()

        def get_finished_spans(self):
            return list(self._spans)


DEFAULT_RESOURCE = Resource.create(
    {
        "service.name": "usb-agents",
        "service.version": "0.1.0",
    }
)

_TRACER_CONFIGURED = False
_METRICS_CONFIGURED = False
_TOOL_LATENCY_HISTOGRAM = None
_STDIO_WAIT_HISTOGRAM = None
_STDIO_ACTIVE_HISTOGRAM = None
_STDIO_AVAILABLE_HISTOGRAM = None


def configure_tracer(
    *,
    exporter: Optional[SpanExporter] = None,
    service_name: str = "usb-agents",
    resource_attrs: Optional[Dict[str, str]] = None,
) -> None:
    """Initialise a tracer provider if one has not been set."""
    global _TRACER_CONFIGURED
    if _TRACER_CONFIGURED:
        return

    if exporter is None:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            except ImportError:  # pragma: no cover - optional dependency
                exporter = ConsoleSpanExporter()
            else:
                headers_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
                headers: Dict[str, str] = {}
                if headers_env:
                    for pair in headers_env.split(","):
                        if "=" in pair:
                            key, value = pair.split("=", 1)
                            headers[key.strip()] = value.strip()
                exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        if exporter is None:
            exporter = InMemorySpanExporter()

    attributes = dict(DEFAULT_RESOURCE.attributes)
    attributes["service.name"] = service_name
    if resource_attrs:
        attributes.update(resource_attrs)

    provider = TracerProvider(resource=Resource.create(attributes))
    if isinstance(exporter, (ConsoleSpanExporter, InMemorySpanExporter)):
        processor = SimpleSpanProcessor(exporter)
    else:
        processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _TRACER_CONFIGURED = True


def configure_metrics(service_name: str = "usb-agents") -> None:
    """Initialise a meter provider when OTLP metrics are configured."""
    global _METRICS_CONFIGURED, _TOOL_LATENCY_HISTOGRAM, _STDIO_WAIT_HISTOGRAM, _STDIO_ACTIVE_HISTOGRAM, _STDIO_AVAILABLE_HISTOGRAM
    if _METRICS_CONFIGURED:
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        _METRICS_CONFIGURED = True
        return

    try:  # pragma: no cover - optional dependency surface
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    except ImportError:
        _METRICS_CONFIGURED = True
        return

    headers_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    headers: Dict[str, str] = {}
    if headers_env:
        for pair in headers_env.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                headers[key.strip()] = value.strip()

    metric_exporter = OTLPMetricExporter(endpoint=endpoint, headers=headers)
    reader = PeriodicExportingMetricReader(metric_exporter)
    resource = Resource.create({**DEFAULT_RESOURCE.attributes, "service.name": service_name})
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("usb_agents")
    _TOOL_LATENCY_HISTOGRAM = meter.create_histogram(
        "usb_agents.tool.latency",
        unit="ms",
        description="Latency of MCP tool invocations in milliseconds.",
    )
    _STDIO_WAIT_HISTOGRAM = meter.create_histogram(
        "usb_agents.stdio.wait_time",
        unit="ms",
        description="Wait time to acquire stdio worker.",
    )
    _STDIO_ACTIVE_HISTOGRAM = meter.create_histogram(
        "usb_agents.stdio.active_workers",
        unit="1",
        description="Active stdio workers at call time.",
    )
    _STDIO_AVAILABLE_HISTOGRAM = meter.create_histogram(
        "usb_agents.stdio.available_workers",
        unit="1",
        description="Available stdio workers at call time.",
    )
    _METRICS_CONFIGURED = True


def record_tool_latency(tool_name: str, latency_ms: float, *, runtime: str, transport: str) -> None:
    configure_metrics()
    if _TOOL_LATENCY_HISTOGRAM is None:
        return
    _TOOL_LATENCY_HISTOGRAM.record(
        latency_ms,
        attributes={
            "tool.name": tool_name,
            "agent.runtime": runtime,
            "agent.transport": transport,
        },
    )


def record_stdio_wait(wait_ms: float) -> None:
    configure_metrics()
    if _STDIO_WAIT_HISTOGRAM is None:
        return
    _STDIO_WAIT_HISTOGRAM.record(wait_ms)


def record_stdio_active(active_count: float) -> None:
    configure_metrics()
    if _STDIO_ACTIVE_HISTOGRAM is None:
        return
    _STDIO_ACTIVE_HISTOGRAM.record(active_count)


def record_stdio_available(available_count: float) -> None:
    configure_metrics()
    if _STDIO_AVAILABLE_HISTOGRAM is None:
        return
    _STDIO_AVAILABLE_HISTOGRAM.record(available_count)


class SpanCapture:
    """Context manager capturing spans in-memory for inspection."""

    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.processor: Optional[SimpleSpanProcessor] = None

    def __enter__(self) -> "SpanCapture":
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            processor = SimpleSpanProcessor(self.exporter)
            provider.add_span_processor(processor)
            self.processor = processor
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider) and self.processor:
            # Remove processor to avoid accumulating exporters between runs.
            multi = getattr(provider, "_active_span_processor", None)
            processors = list(getattr(multi, "_span_processors", []))
            if self.processor in processors:
                processors.remove(self.processor)
                setattr(multi, "_span_processors", tuple(processors))
            self.processor = None

    def finished_spans(self):
        return self.exporter.get_finished_spans()
