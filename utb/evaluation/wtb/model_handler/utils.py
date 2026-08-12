try:
    from tenacity import retry, retry_if_exception_type, wait_random_exponential
except ImportError:
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
            return func

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
