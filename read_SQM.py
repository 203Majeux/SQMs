#!/usr/bin/env python3
"""Read Unihedron SQM-LE instruments and save measurements to CSV.

Configured instruments:
    SQM1 = 192.168.1.167
    SQM2 = 192.168.1.160
    SQM3 = 192.168.1.192

Examples:
    python read_SQM.py SQM3
    python read_SQM.py SQM1 --count 100
    python read_SQM.py SQM2 --interval 5

By default, the script records continuously every 2 seconds.
Stop continuous logging with Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -----------------------------------------------------------------------------
# Instrument configuration: map the names used on the command line to IPs.
# -----------------------------------------------------------------------------
SQM_DEVICES = {
    "SQM1": "192.168.1.167",
    "SQM2": "192.168.1.160",
    "SQM3": "192.168.1.192",
}

DEFAULT_PORT = 10001
DEFAULT_INTERVAL = 2.0
DEFAULT_TIMEOUT = 20.0
DEFAULT_CONNECT_WAIT = 1.5 #1.0


# Typical response from the SQM-LE after sending the `rx` command:
# r, 12.71m,0000000777Hz,0000000000c,0000000.000s, 028.0C
# The regular expression extracts each field while allowing spaces and signs.
RESPONSE_RE = re.compile(
    r"^[rRuU],\s*(?P<mpsas>[+-]?\d+(?:\.\d+)?)m,\s*"
    r"(?P<frequency>\d+)Hz,\s*(?P<counts>\d+)c,\s*"
    r"(?P<period>\d+(?:\.\d+)?)s,\s*"
    r"(?P<temperature>[+-]?\d+(?:\.\d+)?)C"
)


def positive_float(value: str) -> float:
    # argparse passes command-line values as strings, so validate and convert here.
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    # Counts and retry numbers may be zero, but never negative.
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def resolve_device(name: str) -> tuple[str, str]:
    """Return normalized SQM name and its configured IP address."""
    # Accept names such as "sqm1" but use one canonical spelling internally.
    normalized = name.upper()
    if normalized not in SQM_DEVICES:
        valid = ", ".join(SQM_DEVICES)
        raise ValueError(f"Unknown instrument {name!r}. Choose one of: {valid}")
    return normalized, SQM_DEVICES[normalized]


def receive_response(sock: socket.socket, timeout: float) -> bytes:
    """Receive one response, ending at CR/LF or after a short idle period."""
    # Use the long timeout for the first byte and a short timeout after data starts.
    sock.settimeout(timeout)
    data = bytearray()

    while True:
        try:
            chunk = sock.recv(512)
        except socket.timeout:
            # A timeout before any data means the instrument did not answer.
            if data:
                break
            raise

        if not chunk:
            # An empty read means the peer closed the connection.
            break

        data.extend(chunk)
        if b"\r" in chunk or b"\n" in chunk:
            # SQM-LE responses normally end with a line terminator.
            break

        # If there is no terminator, an idle gap marks the end of the response.
        sock.settimeout(0.5)

    return bytes(data)


def request_reading(
    ip: str,
    port: int,
    timeout: float,
    connect_wait: float,
) -> str:
    """Connect to the SQM-LE, send rx, and return the raw response."""
    # The SQM-LE exposes a simple TCP service, normally on port 10001.
    with socket.create_connection((ip, port), timeout=timeout) as sock:
        # Some network/serial adapters need a moment after connection setup.
        time.sleep(connect_wait)
        # `rx` requests one averaged photometric reading.
        sock.sendall(b"rx")
        raw = receive_response(sock, timeout)

    if not raw:
        raise TimeoutError("connected, but the SQM-LE returned no data")

    # The response is ASCII text; remove protocol padding and line endings.
    return raw.decode("ascii", errors="replace").strip("\x00\r\n ")


def parse_reading(raw: str) -> dict[str, object]:
    """Parse a standard SQM-LE rx response."""
    # Reject malformed or unexpected responses before converting values.
    match = RESPONSE_RE.match(raw)
    if not match:
        raise ValueError(f"Unexpected SQM-LE response: {raw!r}")

    values = match.groupdict()
    # Convert text fields into useful Python numeric types for CSV and display.
    return {
        "mpsas": float(values["mpsas"]),
        "frequency_hz": int(values["frequency"]),
        "period_counts": int(values["counts"]),
        "period_s": float(values["period"]),
        "temperature_c": float(values["temperature"]),
        "raw_response": raw,
    }


def append_csv(
    path: Path,
    instrument: str,
    ip: str,
    reading: dict[str, object],
) -> dict[str, object]:
    """Append one measurement to CSV, creating the header if necessary."""
    # Keep these names stable so measurements can be loaded by pandas or Excel.
    fields = [
        "timestamp_utc",
        "instrument",
        "sqm_ip",
        "mpsas",
        "frequency_hz",
        "period_counts",
        "period_s",
        "temperature_c",
        "raw_response",
    ]

    row = {
        # UTC timestamps make measurements from different instruments comparable.
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "instrument": instrument,
        "sqm_ip": ip,
        **reading,
    }

    # Create the destination folder when this is the first measurement.
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0

    # Append rather than overwrite, preserving earlier measurements.
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        csv_file.flush()

    return row


def build_parser() -> argparse.ArgumentParser:
    # Define and document all command-line options in one place.
    device_list = ", ".join(f"{name}={ip}" for name, ip in SQM_DEVICES.items())
    parser = argparse.ArgumentParser(
        description="Continuously read a configured Unihedron SQM-LE and save to CSV.",
        epilog=f"Configured instruments: {device_list}",
    )
    parser.add_argument(
        "instrument",
        help="Instrument name: SQM1, SQM2, or SQM3 (case-insensitive)",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between measurements (default: {DEFAULT_INTERVAL:g})",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT,
        help=f"Network timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--connect-wait",
        type=positive_float,
        default=DEFAULT_CONNECT_WAIT,
        help=f"Delay after TCP connection in seconds (default: {DEFAULT_CONNECT_WAIT:g})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP data port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--count",
        type=nonnegative_int,
        default=0,
        help="Number of measurements; 0 means continue until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: sqm_measurements_<instrument>.csv)",
    )
    parser.add_argument(
        "--retries",
        type=nonnegative_int,
        default=2,
        help="Retries after a failed measurement (default: 2)",
    )
    return parser


def main() -> int:
    # Parse the selected instrument and acquisition settings.
    args = build_parser().parse_args()

    try:
        instrument, ip = resolve_device(args.instrument)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # Store each instrument in its own DATA folder; a supplied output is a file name/path.
    output = (
        Path("DATA") / f"{instrument.lower()}_with_housing" / args.output
        if args.output
        else Path("DATA") / f"{instrument.lower()}_with_housing" / f"sqm_measurements_{instrument}.csv"
    )

    print(f"Instrument: {instrument}")
    print(f"SQM-LE: {ip}:{args.port}")
    print(f"Interval: {args.interval:g} seconds")
    print(f"Timeout: {args.timeout:g} seconds")
    print(f"Saving to: {output.resolve()}")
    if args.count == 0:
        print("Logging continuously. Press Ctrl+C to stop.")
    else:
        print(f"Number of measurements: {args.count}")

    completed = 0

    try:
        # Continue until the requested count is reached, or forever when count is zero.
        while args.count == 0 or completed < args.count:
            cycle_start = time.monotonic()
            last_error: Exception | None = None

            # Retry transient network/protocol failures before giving up this cycle.
            for attempt in range(args.retries + 1):
                try:
                    raw = request_reading(
                        ip=ip,
                        port=args.port,
                        timeout=args.timeout,
                        connect_wait=args.connect_wait,
                    )
                    reading = parse_reading(raw)
                    row = append_csv(output, instrument, ip, reading)
                    completed += 1
                    last_error = None

                    print(
                        f"{row['timestamp_utc']} | {instrument} | "
                        f"{reading['mpsas']:.2f} mpsas | "
                        f"{reading['temperature_c']:.1f} C | "
                        f"{reading['frequency_hz']} Hz"
                    )
                    break

                except (OSError, TimeoutError, ValueError) as exc:
                    last_error = exc
                    if attempt < args.retries:
                        print(
                            f"Measurement failed ({exc}); retrying "
                            f"{attempt + 1}/{args.retries}...",
                            file=sys.stderr,
                        )
                        time.sleep(1)

            if last_error is not None:
                print(f"Measurement error: {last_error}", file=sys.stderr)

            if args.count != 0 and completed >= args.count:
                break

            # Keep the interval measured from the start of each acquisition cycle.
            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.0, args.interval - elapsed))

    except KeyboardInterrupt:
        print("\nLogging stopped by user.")

    print(f"Saved {completed} measurement(s) to {output.resolve()}")
    return 0 if completed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
