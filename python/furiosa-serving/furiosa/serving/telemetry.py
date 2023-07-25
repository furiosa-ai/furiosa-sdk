import logging
import time
from typing import Tuple
from urllib.parse import urlparse

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.span import INVALID_SPAN
from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

INFO = Gauge(
    "fastapi_app_info",
    "FastAPI application information.",
    ["app_name"],
)
REQUESTS = Counter(
    "fastapi_requests_total",
    "Total count of requests by method and path.",
    ["method", "path", "app_name"],
)
RESPONSES = Counter(
    "fastapi_responses_total",
    "Total count of responses by method, path and status codes.",
    ["method", "path", "status_code", "app_name"],
)
REQUESTS_PROCESSING_TIME = Histogram(
    "fastapi_requests_duration_seconds",
    "Histogram of requests processing time by path (in seconds)",
    ["method", "path", "app_name"],
)
EXCEPTIONS = Counter(
    "fastapi_exceptions_total",
    "Total count of exceptions raised by path and exception type",
    ["method", "path", "exception_type", "app_name"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "fastapi_requests_in_progress",
    "Gauge of requests by method and path currently being processed",
    ["method", "path", "app_name"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, app_name: str = "fastapi-app") -> None:
        super().__init__(app)
        self.app_name = app_name
        INFO.labels(app_name=self.app_name).inc()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method = request.method
        path, is_handled_path = self.get_path(request)

        if not is_handled_path:
            return await call_next(request)

        REQUESTS_IN_PROGRESS.labels(method=method, path=path, app_name=self.app_name).inc()
        REQUESTS.labels(method=method, path=path, app_name=self.app_name).inc()
        before_time = time.perf_counter()
        try:
            response = await call_next(request)
        except RuntimeError as e:
            # Workaround for client drops connection
            # https://github.com/encode/starlette/discussions/1527
            # fix in starlette 0.21.0 (https://github.com/encode/starlette/pull/1715)
            if str(e) == 'No response returned.' and await request.is_disconnected():
                status_code = 499
                return Response(status_code=status_code)
            raise e
        except BaseException as e:
            status_code = HTTP_500_INTERNAL_SERVER_ERROR
            EXCEPTIONS.labels(
                method=method, path=path, exception_type=type(e).__name__, app_name=self.app_name
            ).inc()
            raise e from None
        else:
            status_code = response.status_code
            after_time = time.perf_counter()
            # retrieve trace id for exemplar
            span = trace.get_current_span()
            if span == INVALID_SPAN:
                exemplar = {}
            else:
                trace_id = trace.format_trace_id(span.get_span_context().trace_id)
                exemplar = {'TraceID': trace_id}

            REQUESTS_PROCESSING_TIME.labels(
                method=method, path=path, app_name=self.app_name
            ).observe(after_time - before_time, exemplar=exemplar)
        finally:
            RESPONSES.labels(
                method=method, path=path, status_code=status_code, app_name=self.app_name
            ).inc()
            REQUESTS_IN_PROGRESS.labels(method=method, path=path, app_name=self.app_name).dec()

        return response

    @staticmethod
    def get_path(request: Request) -> Tuple[str, bool]:
        for route in request.app.routes:
            match, child_scope = route.matches(request.scope)
            if match == Match.FULL:
                return route.path, True

        return request.url.path, False


def metrics(request: Request) -> Response:
    return Response(generate_latest(REGISTRY), headers={"Content-Type": CONTENT_TYPE_LATEST})


def setup_metrics(app: FastAPI, app_name: str, metric_path: str = "/metrics") -> None:
    app.add_middleware(PrometheusMiddleware, app_name=app_name)
    app.add_route(metric_path, metrics)


def setup_otlp(app: FastAPI, app_name: str, endpoint: str, log_correlation: bool = True) -> None:
    # Setting OpenTelemetry
    # set the service name to show in traces
    resource = Resource.create(attributes={"service.name": app_name, "app": app_name})

    # set the tracer provider
    tracer = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer)

    def is_valid_url(endpoint: str) -> bool:
        try:
            url = urlparse(endpoint)
            return all([url.scheme, url.netloc])
        except ValueError:
            return False

    if is_valid_url(endpoint):
        tracer.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        setup_logger(True)
    else:
        setup_logger(False)
        if endpoint is not None:
            logging.warning(f"Invalid OpenTelemetry Endpoint URL: {endpoint}")

    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer)


def setup_logger(otlp_enabled: bool) -> None:
    if otlp_enabled:
        logging_format = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s] - %(message)s"  # noqa: E501
    else:
        logging_format = (
            "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
        )

    LoggingInstrumentor().instrument(
        set_logging_format=True,
        logging_format=logging_format,
    )
