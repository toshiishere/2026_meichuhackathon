# Demo playback and recording recovery

## CSI inputs: important limitation

Deployment currently selects **one receiver**, not three concurrent inputs.
Training reads all recorded receivers, but `preprocess.py` creates independent
single-link examples (`receiver_mode: shared single-link backbone; each receiver
is an example`). The current checkpoint is not a fused three-input model.
This update does not change that model contract or introduce score fusion.

## Deployment video

Recorded-session replay automatically plays `raw/video.mp4`, using
`raw/video_frames.parquet` to map the CSI replay clock to video PTS. Buffered
phone recordings use `capture_timestamp_ns`, with legacy recordings falling
back to `host_timestamp_ns`. Replay speed applies to both CSI and video.
Leaving and reopening Deploy reconnects playback to the worker's playhead;
stopping, completion and service disconnection pause playback. Video is muted
and controlled by deployment, not by an independent video seek bar.

For live sources, choose a live camera. The preview selects the buffered camera
frame nearest the current CSI capture timestamp (maximum difference 500 ms)
rather than displaying an unrelated latest frame. Camera failure does not stop
CSI inference. Preview refresh and inference have normal network/window latency;
this is timestamp-aligned visualization, not a hard real-time guarantee.

## Delete labels / model

Train → Results has separate **Delete labels** and **Delete model** buttons with
confirmation. These delete published artifacts and earlier run outputs of the
selected type. Raw recordings, job logs/options and `labels_used.csv` provenance
snapshots are preserved. Model deletion includes session-local checkpoint and
pretrained copies, not the project's shared pretrained backbone. The actions
are blocked while training/deployment or another operation owns the session.
Deleting labels leaves the existing model available but marks it stale.

## Recover incomplete sessions

Sessions → **Recover** validates closed video, its frame index and CSI files,
then renames `.filename.tmp` (or `filename.tmp`) to `filename`. Conflicting,
empty, unreadable or missing artifacts are rejected before any rename. Recovery
does not repair damaged/unclosed containers and never overwrites a final file.
Already-finalized files from partial finalization are supported.

Recovered sessions have `status: complete` for training/replay compatibility,
plus `recovered: true`, `quality: recovered`, a recovery timestamp and a rename
list. The UI labels them **recovered**. Original errors and acquisition
statistics remain intact: recovered does not mean good-quality data. Active
acquisition threads and busy sessions cannot be recovered. No existing session
is recovered or deleted automatically.
