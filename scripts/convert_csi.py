#!/usr/bin/env python3
"""Convert recorded .csv.zst or raw CSI binary v1 to CSV; preview 10 lines."""

import argparse
from contextlib import contextmanager
import csv
import io
from itertools import islice
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apps.hardware_service.app.csi import FIELDS, parse_csi
from apps.hardware_service.app.csi_wire import MAGIC, SerialFramer


@contextmanager
def open_input(path):
    with path.open("rb") as source:
        compressed = source.read(4) == b"\x28\xb5\x2f\xfd" or path.suffix == ".zst"
        source.seek(0)
        if not compressed:
            yield source
            return
        try:
            import zstandard
        except ImportError:
            command = shutil.which("zstd")
            if not command:
                raise ValueError(
                    "Reading .zst requires the zstd command or: pip install zstandard"
                ) from None
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(
                    [command, "-dc", "--", str(path)],
                    stdout=subprocess.PIPE,
                    stderr=errors,
                )
                try:
                    yield process.stdout
                    if process.wait() != 0:
                        errors.seek(0)
                        raise ValueError(errors.read(4096).decode(errors="replace").strip())
                finally:
                    process.stdout.close()
                    if process.poll() is None:
                        process.terminate()
                    process.wait()
        else:
            try:
                with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                    yield reader
            except zstandard.ZstdError as error:
                raise ValueError(f"Cannot decompress CSI input: {error}") from error


def convert(source, destination, input_format):
    """Stream records, preserving samples and any recorded host timestamps."""
    count, ignored = 0, 0
    with io.BufferedReader(source) as buffered:
        if input_format == "auto":
            prefix = buffered.peek(128).lstrip(b"\xef\xbb\xbf")
            input_format = "csv" if prefix.startswith(b"host_timestamp_ns,") else "binary"
        if input_format == "csv":
            csv.field_size_limit(16 * 1024 * 1024)
            with io.TextIOWrapper(buffered, encoding="utf-8-sig", newline="") as text:
                reader = csv.DictReader(text)
                if not reader.fieldnames or not {"data", "tx_seq"}.issubset(reader.fieldnames):
                    raise ValueError("Expected a recorded CSI CSV header with data and tx_seq columns")
                writer = csv.DictWriter(destination, reader.fieldnames, lineterminator="\n")
                writer.writeheader()
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"Malformed CSV record near line {reader.line_num}")
                    writer.writerow(row)
                    count += 1
        else:
            writer = csv.DictWriter(destination, FIELDS, lineterminator="\n")
            writer.writeheader()
            framer = SerialFramer()

            def write_record(raw):
                nonlocal count, ignored
                record = parse_csi(raw)  # Includes binary length, metadata and CRC checks.
                if record is None:
                    ignored += bool(raw.strip())
                    return
                writer.writerow(record)
                count += 1

            while chunk := buffered.read(64 * 1024):
                for raw, _ in framer.feed(chunk, 0):
                    write_record(raw)
            if framer.buffer:
                remaining = bytes(framer.buffer)
                if remaining.startswith(MAGIC) or MAGIC.startswith(remaining):
                    raise ValueError("Truncated binary CSI frame at end of input")
                write_record(remaining)
    if not count:
        raise ValueError("No CSI records found in the input")
    return count, ignored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Recorded .csv.zst/.csv or raw binary CSI capture")
    parser.add_argument("-o", "--output", type=Path, help="CSV destination (default: INPUT stem.csv in current directory)")
    parser.add_argument("--input-format", choices=["auto", "csv", "binary"], default="auto")
    parser.add_argument("--preview-lines", type=int, default=10, help="Lines printed to stdout, including header (default: 10)")
    args = parser.parse_args(argv)
    if args.preview_lines < 0:
        parser.error("--preview-lines must be nonnegative")
    name = args.input.name.removesuffix(".zst")
    output = args.output or Path(Path(name).stem + ".csv")
    temporary = None
    try:
        if args.input.resolve() == output.resolve():
            raise ValueError("Output must differ from input; choose a new path with --output")
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists: {output}; choose a new path with --output")
        # Publish only a complete conversion, without replacing existing files.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as destination:
            temporary = Path(destination.name)
            with open_input(args.input) as source:
                count, ignored = convert(source, destination, args.input_format)
        os.link(temporary, output)
        print(f"Wrote {count} CSI records to {output} (ignored {ignored} non-CSI lines).", file=sys.stderr)
        with output.open(encoding="utf-8", newline="") as readable:
            sys.stdout.writelines(islice(readable, args.preview_lines))
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
