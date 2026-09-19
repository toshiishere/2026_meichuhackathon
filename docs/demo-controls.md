# Demo playback and recording recovery

## CSI inputs: three receivers, one pose

Deployment reads **every selected receiver** (all of them by default), builds one
window per receiver ending at the same timestamp, scores them in a single batched
forward pass and averages the class probabilities into one pose. Training feeds
each receiver's window for a labeled interval to the same single-link backbone
(`receiver_mode: shared single-link backbone; each receiver is an example`), so
this is the deployment counterpart of that: score fusion, not a retrained
multi-link model, and the existing checkpoints stay valid.

The shared window end is the newest timestamp every live receiver covers, so the
links are compared at the same moment rather than at whatever each last sent. A
receiver silent for over 500 ms (scaled by replay speed) drops out of the fusion
and stops holding the shared clock back; a receiver whose window lacks coverage
is reported as uncovered. The UI shows the fused pose, which receivers were
fused, and each receiver's own score before fusion.

## Deployment video

Recorded-session replay automatically plays `raw/video.mp4`, using
`raw/video_frames.parquet` to map the CSI replay clock to video PTS. Every
replayed receiver shares that one clock, so video, CSI and the fused window all
advance together. Buffered phone recordings use `capture_timestamp_ns`, with
legacy recordings falling back to `host_timestamp_ns`. Replay speed applies to
both CSI and video. Leaving and reopening Deploy reconnects playback to the
worker's playhead; stopping, completion and service disconnection pause
playback. Video is muted and controlled by deployment, not by an independent
video seek bar.

A status poll is already a round trip old when it arrives, so the page does not
simply seek to the value it received. It extrapolates the worker's playhead from
the sample time (minus half the measured round trip) at the replay speed, then
absorbs drift up to 350 ms by trimming playback rate by at most 10% and only
seeks beyond that. The panel reports the current video-to-playhead offset in
milliseconds. While paused, stopped or completed, the video is placed exactly on
the reported playhead.

For live sources, choose a live camera. The preview is fetched for the **fused
CSI clock** — the shared window end across live receivers — and the collector
returns the buffered frame nearest that time, refusing anything further than
500 ms away rather than showing an unrelated latest frame. The panel reports the
served frame's real offset from that CSI time, taken from the collector's
`X-Host-Timestamp-Ns` header, so the alignment is visible rather than assumed.
Camera and CSI timestamps come from the same monotonic clock in the collector
process, including buffered phone frames, which are converted to that clock
before use. The camera thread now captures at the camera's own frame rate and
keeps several seconds of frames, so alignment is limited by the camera's frame
interval, not by a fixed preview throttle. Camera failure does not stop CSI
inference. Preview refresh and inference have normal network/window latency;
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
