"""
Decode raw ESP32 binary_v1 CSI packets (as captured by our session collector)
into L-LTF amplitude -- the same 52-subcarrier legacy measurement ESP-Fi HAR's
dataset/model uses.

The L-LTF's position within the raw int8 sample array depends on the WiFi
bandwidth the packet was captured at -- HT20 and HT40 use DIFFERENT index
ranges, even though both give you back 52 subcarriers. Verified against:
- espressif/esp-csi#146 (HT40 no-STBC, 384-byte payload: indices 6-31, 33-58)
- espressif/esp-csi#146 (HT20 no-STBC, 256-byte payload: indices 1-26, 38-63)

Only non-STBC packets are supported (this session collector's captures have
all been stbc=0 so far). STBC changes the frame layout and would need its own
index table -- check the `stbc` column before trusting this decoder on new
data, `main()` below does that check automatically and raises if it sees any.
"""
import ast

import numpy as np
import pandas as pd

LLTF_IDX_HT20 = list(range(1, 27)) + list(range(38, 64))   # 256-byte, no-STBC
LLTF_IDX_HT40 = list(range(6, 32)) + list(range(33, 59))   # 384-byte, no-STBC

BANDWIDTH_TABLE = {
    0: ("HT20", LLTF_IDX_HT20),
    1: ("HT40", LLTF_IDX_HT40),
}


def lltf_indices_for(bandwidth_code):
    if bandwidth_code not in BANDWIDTH_TABLE:
        raise ValueError(
            f"Unknown bandwidth code {bandwidth_code!r} -- only HT20(0)/HT40(1) "
            f"index tables are verified. Don't guess; check espressif/esp-csi "
            f"issue #146 for this packet size before adding a new entry."
        )
    return BANDWIDTH_TABLE[bandwidth_code]


def decode_amplitude(data_str, lltf_idx):
    """data_str: the CSV 'data' column, a stringified list of raw int8 samples,
    interleaved as imaginary,real,imaginary,real,...
    Returns amplitude for the 52 L-LTF complex pairs, shape (52,).
    """
    arr = np.array(ast.literal_eval(data_str), dtype=np.float32)
    imag = arr[0::2]
    real = arr[1::2]
    complex_vals = real + 1j * imag
    lltf = complex_vals[lltf_idx]
    return np.abs(lltf)


def load_csi_csv(path):
    """Load a raw_csi_<receiver>.csv (or decompressed .csv.zst), and determine
    which bandwidth's index table applies. Raises if the file mixes bandwidths
    or contains any STBC packets, rather than silently decoding some of it wrong.
    """
    df = pd.read_csv(path, usecols=["host_timestamp_ns", "bandwidth", "stbc", "len", "data"])

    if (df["stbc"] != 0).any():
        raise ValueError(
            f"{path}: found STBC packets (stbc != 0) -- this decoder only "
            f"supports non-STBC index tables. Do not decode these rows without "
            f"first finding the correct STBC index range."
        )

    bw_codes = df["bandwidth"].unique()
    if len(bw_codes) != 1:
        raise ValueError(
            f"{path}: mixed bandwidth codes {bw_codes} in one file -- decode "
            f"per-bandwidth subset separately, don't apply one index table to all rows."
        )

    bw_name, lltf_idx = lltf_indices_for(int(bw_codes[0]))
    df = df.sort_values("host_timestamp_ns").reset_index(drop=True)
    return df, bw_name, lltf_idx


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    df, bw_name, lltf_idx = load_csi_csv(path)
    print(f"{path}: {len(df)} rows, bandwidth={bw_name}, L-LTF indices={lltf_idx[0]}..{lltf_idx[-1]} ({len(lltf_idx)} subcarriers)")
    amp = decode_amplitude(df["data"].iloc[0], lltf_idx)
    print(f"first row amplitude: shape={amp.shape}, min={amp.min():.2f}, max={amp.max():.2f}, mean={amp.mean():.2f}")
