import pytest
from apps.hardware_service.app.jobs import Jobs


def test_persistence_failure_does_not_leak_hardware_lease(tmp_path, monkeypatch):
    jobs = Jobs(tmp_path)
    def fail(*args):
        raise OSError('storage full')
    monkeypatch.setattr(jobs.db, 'put', fail)
    with pytest.raises(OSError, match='storage full'):
        jobs.submit('probe', lambda log, stop: {})
    assert not jobs.lease.locked()
