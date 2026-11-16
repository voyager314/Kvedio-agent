"""Failures the store reports to its callers."""


class StorageError(Exception):
    """The store cannot serve the request: an unknown row, an illegal graph, or a damaged file.

    The message names what was wrong in terms of the caller's own values (a video
    id, a task id), so it can be shown without translation.
    """
