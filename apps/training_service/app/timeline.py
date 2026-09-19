"""Video PTS to collector capture clock, including buffered phone recordings."""

import numpy as np
import pyarrow.parquet as pq


class Timeline:
    def __init__(self, path):
        table = pq.read_table(path).to_pydict()
        self.pts = np.asarray(table["video_pts_s"], dtype=np.float64)
        self.clock_column = (
            "capture_timestamp_ns"
            if "capture_timestamp_ns" in table
            else "host_timestamp_ns"
        )
        stamps = table[self.clock_column]
        if any(s is None for s in stamps):
            raise ValueError("Video frame capture timestamps are missing")
        self.ns = np.asarray(stamps, dtype=np.int64)
        if (
            len(self.pts) < 2
            or not np.isfinite(self.pts).all()
            or np.any(np.diff(self.pts) <= 0)
            or np.any(np.diff(self.ns) <= 0)
            or list(table["frame_idx"]) != list(range(len(self.pts)))
        ):
            raise ValueError(
                "Video frame index must contain ordered, unique frame/capture timestamps"
            )
        self.relative_ns = self.ns - self.ns[0]

    def host_ns(self, seconds):
        if not self.pts[0] <= seconds <= self.pts[-1]:
            raise ValueError("Label time falls outside the recorded video timeline")
        return int(self.ns[0]) + round(
            float(np.interp(seconds, self.pts, self.relative_ns))
        )

    def video_seconds(self, host_ns):
        """Map the CSI capture clock back to video PTS (also for buffered phones)."""
        return float(np.interp(host_ns - int(self.ns[0]), self.relative_ns, self.pts))

    def contiguous(self, start, end, max_gap=0.5):
        left = max(0, np.searchsorted(self.pts, start, side="right") - 1)
        right = min(len(self.pts), np.searchsorted(self.pts, end, side="left") + 1)
        return not np.any(np.diff(self.pts[left:right]) > max_gap)
