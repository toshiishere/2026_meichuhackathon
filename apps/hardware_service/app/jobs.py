import threading
import uuid
from apps.common.storage import Registry, utc_now


class BusyError(RuntimeError):
    pass


class Jobs:
    def __init__(self, root):
        self.root = root
        self.db = Registry(root / "app/hardware.sqlite")
        self.lease = threading.Lock()
        self.current = None
        self.cancel_event = None
        for job in self.db.list("jobs"):
            if job["status"] in {"queued", "running"}:
                job.update(
                    status="failed",
                    error="Hardware service restarted during operation",
                    finished_at=utc_now(),
                )
                self.db.put("jobs", job["id"], job)

    def acquire(self):
        if not self.lease.acquire(blocking=False):
            raise BusyError(
                "Hardware is busy. Stop recording or close camera preview; wait for the current hardware job to finish."
            )

    def submit(self, kind, function):
        self.acquire()
        jid = uuid.uuid4().hex
        log_path = self.root / "app/job_logs" / (jid + ".log")
        job = dict(
            id=jid,
            kind=kind,
            status="queued",
            created_at=utc_now(),
            result=None,
            error=None,
        )
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.db.put("jobs", jid, job)
        except Exception:
            self.lease.release()
            raise
        stop = threading.Event()
        self.current, self.cancel_event = jid, stop

        def work():
            try:
                job.update(status="running", started_at=utc_now())
                self.db.put("jobs", jid, job)
                with log_path.open("a", buffering=1) as f:

                    def log(line):
                        f.write(f"{utc_now()} {line}\n")

                    log(f"Starting {kind}")
                    result = function(log, stop)
                    job.update(
                        result=result,
                        status="cancelled"
                        if stop.is_set() and kind != "collection"
                        else "completed",
                    )
                    log(f"Finished {kind}: {job['status']}")
            except Exception as e:
                job.update(
                    status="cancelled"
                    if stop.is_set() and kind != "collection"
                    else "failed",
                    error=str(e),
                )
                with log_path.open("a") as f:
                    f.write(f"{utc_now()} ERROR: {e}\n")
            finally:
                job["finished_at"] = utc_now()
                try:
                    self.db.put("jobs", jid, job)
                finally:
                    self.current = self.cancel_event = None
                    self.lease.release()

        threading.Thread(target=work, name=f"job-{jid}", daemon=True).start()
        return job.copy()

    def cancel(self, jid):
        job = self.db.get("jobs", jid)
        if not job:
            raise KeyError(jid)
        if jid == self.current and self.cancel_event:
            if job["kind"] == "flash":
                raise ValueError(
                    "Firmware jobs cannot be cancelled during a flash; wait for completion"
                )
            self.cancel_event.set()
        return job
