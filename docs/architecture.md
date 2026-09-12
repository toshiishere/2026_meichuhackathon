# Collection architecture

Scope: prompt milestones 1–6 only. The present deliverable is synchronized raw
CSI and camera collection and inspection. No labeling, learning, inference,
window datasets, or downstream feature processing is implemented.

Browser (React/TypeScript) → nginx → FastAPI backend → internal hardware API.
Only the hardware service can access devices. Backend stores logical assignments
and hardware service stores jobs in separate SQLite databases. Session directories
are canonical; the session index and root Parquet manifest are rebuildable.

The collector uses one serial acquisition thread and bounded queue per receiver,
one camera acquisition thread and bounded queue, separate streaming file writers,
and one shared monotonic start gate. Readers timestamp at complete serial-line
receipt / successful camera read, before queueing. The host clock synchronizes
modalities; transmitter sequence numbers synchronize receivers. USB buffering and
camera driver latency remain measurement limitations, not corrected estimates.

Shutdown stops readers, drains queues, closes all writers, checks output counts,
and atomically renames completed files. A failed writer or empty stream produces
an incomplete session. Overflows are counted and mark collection degraded.
Startup finds interrupted metadata and marks it incomplete without deleting data.
CSI compressed streams periodically flush independent frames for crash inspection;
fragmented MP4 preserves already flushed fragments. Temporary artifacts are kept.
Raw files are never overwritten by a new session; names are reserved using mkdir.

Hardware access has a service-wide lease: tests, probe, builds/flashes, and recording
cannot overlap. Camera preview owns the lease until its stream closes. Recording
runs outside HTTP request handling. The backend provides SSE status and job updates
once per second; it does not handle acquisition packets.

Firmware was inspected before implementation: existing local source has upstream
commit 8633d67152db2808f141cc1595970aa9cf406045, generated SDK configs identify IDF
5.5.0, sender target esp32, receiver esp32c3. Sender and receiver app_main.c match
upstream examples. Vendored project copies retain SDK configs, dependency locks,
100 Hz, channel 11, HT40 and RX 921600 baud. Original tree and boards are untouched.
Firmware cache keys include source content, original upstream commit, target,
IDF version and blink configuration. Builds use private copies of the projects.
