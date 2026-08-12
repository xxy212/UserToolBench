import random
import time

try:
    from tenacity import retry, retry_if_exception_type, wait_random_exponential
except ModuleNotFoundError:
    retry = None
    retry_if_exception_type = None
    wait_random_exponential = None


def retry_with_backoff(error_type, min_wait=6, max_wait=120, **kwargs):
    """
    General decorator to retry with backoff for a specific error type.

    :param error_type: The exception type to retry on.
    :param min_wait: Minimum wait time for the backoff.
    :param max_wait: Maximum wait time for the backoff.
    """

    def decorator(func):
        if retry is None:
            stop = kwargs.get("stop")
            max_attempts = getattr(stop, "max_attempt_number", 6) if stop is not None else 6

            def wrapped(*args, **kwargs):
                last_error = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except error_type as exc:
                        last_error = exc
                        if attempt == max_attempts:
                            break
                        sleep_seconds = min(max_wait, min_wait * (2 ** (attempt - 1))) + random.random()
                        print(
                            f"Attempt {attempt} failed. Sleeping for {float(round(sleep_seconds, 2))} seconds before retrying..."
                            f"Error: {exc}"
                        )
                        time.sleep(sleep_seconds)
                raise last_error

            return wrapped

        @retry(
            wait=wait_random_exponential(min=min_wait, max=max_wait),
            retry=retry_if_exception_type(error_type),
            before_sleep=lambda retry_state: print(
                f"Attempt {retry_state.attempt_number} failed. Sleeping for {float(round(retry_state.next_action.sleep, 2))} seconds before retrying..."
                f"Error: {retry_state.outcome.exception()}"
            ),
            **kwargs,
        )
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped

    return decorator
