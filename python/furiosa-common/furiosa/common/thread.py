import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
import functools
from functools import wraps
from typing import Any, Callable


def synchronous(f: Callable) -> Callable:
    """
    Run async function in place and return the result
    """

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()

            # If running event loop already exists, run the coroutine within a executor
            executor = ThreadPoolExecutor()
            future = executor.submit(asyncio.run, f(*args, **kwargs))
            return future.result()
        except RuntimeError:
            # When no running event loop exists, create new one and run the coroutine
            return asyncio.run(f(*args, **kwargs))

    return wrapper


def asynchronous(f: Callable) -> Callable:
    """
    Replace sync function to async using aysncio thread pool
    """

    @wraps(f)
    async def wrapper(*args: Any, **kwargs: Any):
        return await _to_thread(f, *args, **kwargs)

    return wrapper


async def _to_thread(func, *args, **kwargs):
    """
    Copied from asyncio.to_thread() in Python 3.9
    """
    loop = asyncio.events.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)
