"""Failures reported by model endpoints."""


class ModelError(Exception):
    """A model call could not be completed.

    The message names the status and the provider's own message when it sent one.
    It never repeats the request headers, the credential, or the payload: those
    are the three places a secret can leak into a log line.
    """
