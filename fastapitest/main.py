"""
FastAPI demo: sync vs async endpoints + middleware
Run:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
Test: Postman collection (see postman_collection.json) or curl
"""

import time
import asyncio
import uuid
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("demo")

app = FastAPI(title="Sync vs Async vs Middleware Demo")


# ----------------------------------------------------------------------
# MIDDLEWARE
# ----------------------------------------------------------------------
# Middleware wraps EVERY request. Order matters: middlewares added later
# run "closer" to the request (i.e. last-added = first-executed on the
# way in, last-executed on the way out) because FastAPI/Starlette wraps
# them like an onion.

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attaches a unique request ID to every request/response (common in real systems
    for tracing a request across logs, e.g. in your RAG pipeline logs)."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def timing_logger_middleware(request: Request, call_next):
    """Logs method, path, status code, and how long the request took."""
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    request_id = getattr(request.state, "request_id", "n/a")

    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"-> {response.status_code} ({duration_ms:.1f}ms)"
    )
    response.headers["X-Process-Time-ms"] = f"{duration_ms:.1f}"
    return response


@app.middleware("http")
async def auth_check_middleware(request: Request, call_next):
    """A toy auth middleware: requires header 'X-API-Key: secret123' for /secure/* routes.
    Demonstrates short-circuiting a request before it reaches the endpoint."""
    if request.url.path.startswith("/secure"):
        api_key = request.headers.get("X-API-Key")
        if api_key != "secret123":
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid X-API-Key header"},
            )
    return await call_next(request)


# ----------------------------------------------------------------------
# SYNC vs ASYNC ENDPOINTS
# ----------------------------------------------------------------------
# Key idea to test in Postman: hit /sync/slow and /async/slow concurrently
# (open 3-4 requests at once in different Postman tabs, or use Postman
# Runner) and watch the timing logs in your terminal.
#
# - def (sync) endpoints run in FastAPI's threadpool -> they don't block
#   the event loop, but you're limited by threadpool size.
# - async def endpoints run directly on the event loop -> if you use
#   blocking calls (time.sleep) inside them, you WILL block the whole
#   event loop and every other concurrent request.
# - async def with truly async calls (asyncio.sleep, async DB drivers,
#   httpx.AsyncClient) lets many requests be handled concurrently on a
#   single thread.

@app.get("/sync/fast")
def sync_fast():
    """Plain sync endpoint, no blocking work."""
    return {"type": "sync", "speed": "fast", "message": "instant response"}


@app.get("/sync/slow")
def sync_slow():
    """Sync endpoint with blocking I/O simulation (time.sleep).
    FastAPI runs this in a threadpool, so it does NOT block other
    concurrent requests by itself -- but threadpool has limited workers
    (default 40), so under heavy load this can become a bottleneck."""
    time.sleep(3)  # simulates blocking call: sync DB driver, sync requests.get(), etc.
    return {"type": "sync", "speed": "slow", "message": "blocked for 3s (in threadpool)"}


@app.get("/async/fast")
async def async_fast():
    """Plain async endpoint, no blocking work."""
    return {"type": "async", "speed": "fast", "message": "instant response"}


@app.get("/async/slow")
async def async_slow():
    """Properly async endpoint using asyncio.sleep (non-blocking).
    This frees the event loop while 'waiting', so other requests
    (even other calls to this same endpoint) get handled concurrently."""
    await asyncio.sleep(3)  # simulates async I/O: async DB driver, httpx.AsyncClient, etc.
    return {"type": "async", "speed": "slow", "message": "awaited for 3s (event loop free)"}


@app.get("/async/bad-blocking")
async def async_bad_blocking():
    """ANTI-PATTERN: an async def endpoint that uses a BLOCKING call
    (time.sleep) instead of an async one. This blocks the entire event
    loop -- every other request (even on unrelated endpoints) stalls
    until this finishes. Great one to demo the failure mode in Postman:
    fire this + /async/fast at the same time and watch /async/fast hang."""
    time.sleep(3)  # WRONG: blocking call inside async def
    return {"type": "async", "speed": "bad", "message": "blocked the whole event loop for 3s"}


# ----------------------------------------------------------------------
# SECURE ROUTES (to demonstrate middleware short-circuit)
# ----------------------------------------------------------------------

@app.get("/secure/data")
async def secure_data():
    return {"message": "you passed the auth_check_middleware", "data": [1, 2, 3]}


# ----------------------------------------------------------------------
# MISC: request_id available inside route via request.state
# ----------------------------------------------------------------------

@app.get("/whoami")
async def whoami(request: Request):
    return {"your_request_id": request.state.request_id}


@app.get("/")
def root():
    return {
        "message": "FastAPI sync/async/middleware demo",
        "try": [
            "GET /sync/fast",
            "GET /sync/slow   (3s, threadpool)",
            "GET /async/fast",
            "GET /async/slow  (3s, non-blocking)",
            "GET /async/bad-blocking (3s, BLOCKS event loop - anti-pattern demo)",
            "GET /secure/data (needs header X-API-Key: secret123)",
            "GET /whoami      (shows request id set by middleware)",
        ],
    }