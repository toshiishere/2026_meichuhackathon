"""Cross-container exclusion for training and session removal on shared storage."""

import fcntl
from contextlib import contextmanager


@contextmanager
def session_lock(root, session_id):
    folder = root / "app/session_locks"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{session_id}.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Session is busy training, deploying or being removed; wait for completion"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
