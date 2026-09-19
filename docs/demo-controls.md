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

### Why a recording needs an index before it can play

The collector writes fragmented MP4 (`frag_keyframe+empty_moov`) so a killed
recording still holds every finished fragment. A fragmented file carries no
sample table, so unless it also carries a segment index a browser must read the
**whole** file before it can report a duration, show a frame or seek: measured
in Chrome against a 17-minute, 258 MB session, nothing appeared for ~110
seconds, while an indexed copy of the same recording showed metadata in 18 ms
and seeked across 10 minutes in 15 ms.

Recordings now carry that index: the collector adds `global_sidx`, written on
close, which leaves a killed recording exactly as recoverable as before (its
fragments survive; only the index is missing). Older recordings are indexed on
demand instead — the first replay deployment stream-copies the video, with no
re-encode, into `derived/video.mp4` and reuses it afterwards. `raw/` is never
modified, and the copy is rebuilt if the recording ever changes. Deployment
status reports which file the page should play (`video_path`), and
`video_status: preparing` while the copy is being made. If indexing fails,
replay falls back to the raw recording and says that playback may start slowly.

Recorded-session replay automatically plays that video, using
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

## Fall alerts to Discord

`make up` starts a `dc-bot` service alongside the collector. It holds a Discord
gateway session, which is what makes a bot appear **online** — REST calls alone
leave it grey — and exposes `POST /alert` on the internal network for the
deployment worker, plus `GET /health` reporting whether it is connected. The
token comes from `DISCORD_BOT_TOKEN` in `.env` (never from a source file) and
`DISCORD_CHANNEL_ID` overrides the default channel. A token Discord rejects is
reported once and not retried, because retrying a rejected token earns a rate
limit; fix `.env` and restart the service. Changing the token requires
recreating the container (`make up` does this).

Deploy has a **Send a Discord alert when a fall is detected** switch. It can be
turned on before starting and flipped during a run; turning it on asks the bot
whether it is online and says so immediately rather than staying silent until a
fall happens. Falls are always detected and listed in the Deploy panel — the
switch only decides whether anything leaves this machine.

The rule, applied to the fused pose predictions:

- a prediction labelled **Falling** starts a four-second watch;
- inside that watch, stillness is the span from the first **Static** prediction
  after the fall to the latest one, and must reach two seconds. A stray
  prediction of another action in between does not cancel it; a second fall
  restarts the watch;
- one alert per fall, with a 30-second cooldown.

Because interruptions do not reset the span, a couple of noisy predictions
between two Static ones still count as stillness — deliberately tolerant, so a
real fall is not missed for one flickering window.

Every one of those times is a **source** time — seconds along the live capture,
or along the recording being replayed. Replay speed is not involved, so a
recording replayed at 4x raises the same alerts, at the same points of the
recording, as it would at 1x. Recorded replays do alert, and their message says
which session was replayed so nobody reads a demo as a live emergency. The
message itself is the bot's existing wording, with the detection detail
appended. A failed or refused send is reported in Deploy and never interrupts
inference.
