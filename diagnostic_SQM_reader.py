#!/usr/bin/env python3
"""Diagnostic reader and CSV logger for a Unihedron SQM-LE."""

from __future__ import annotations

import argparse
import csv
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RESPONSE_RE = re.compile(
    r"^[rRuU],\s*(?P<mpsas>[+-]?\d+(?:\.\d+)?)m,\s*"
    r"(?P<frequency>\d+)Hz,\s*(?P<counts>\d+)c,\s*"
    r"(?P<period>\d+(?:\.\d+)?)s,\s*"
    r"(?P<temperature>[+-]?\d+(?:\.\d+)?)C"
)

# First try the protocol exactly as documented. The alternatives help with
# XPort configurations that expect a line terminator before forwarding data.
COMMAND_VARIANTS = (b"rx", b"rx\r", b"rx\n", b"rx\r\n")


def receive_available(sock: socket.socket, first_timeout: float, idle_timeout: float = 0.5) -> bytes:
    """Wait for the first byte, then collect data until CR/LF or an idle gap."""
    sock.settimeout(first_timeout)
    data = bytearray()
    while True:
        try:
            chunk = sock.recv(512)
        except socket.timeout:
            if data:
                break
            raise
        if not chunk:
            break
        data.extend(chunk)
        if b"\r" in chunk or b"\n" in chunk:
            break
        sock.settimeout(idle_timeout)
    return bytes(data)


def exchange(ip: str, port: int, command: bytes, timeout: float, connect_wait: float) -> str:
    with socket.create_connection((ip, port), timeout=timeout) as sock:
        # Some serial device servers need a short interval after TCP connect.
        time.sleep(connect_wait)
        sock.sendall(command)
        raw = receive_available(sock, timeout)
    if not raw:
        raise TimeoutError("connected, but the instrument returned no bytes")
    return raw.decode("ascii", errors="replace").strip("\x00\r\n ")


def auto_read(ip: str, port: int, timeout: float, connect_wait: float, verbose: bool):
    errors = []
    for command in COMMAND_VARIANTS:
        try:
            if verbose:
                print(f"Trying command bytes: {command!r}")
            raw = exchange(ip, port, command, timeout, connect_wait)
            match = RESPONSE_RE.match(raw)
            if not match:
                errors.append(f"{command!r}: unexpected response {raw!r}")
                continue
            g = match.groupdict()
            return {
                "mpsas": float(g["mpsas"]),
                "frequency_hz": int(g["frequency"]),
                "period_counts": int(g["counts"]),
                "period_s": float(g["period"]),
                "temperature_c": float(g["temperature"]),
                "command_bytes": repr(command),
                "raw_response": raw,
            }
        except (OSError, TimeoutError) as exc:
            errors.append(f"{command!r}: {exc}")
    raise RuntimeError("No command variant produced a reading:\n  " + "\n  ".join(errors))


def save(path: Path, ip: str, reading: dict):
    fields = ["timestamp_utc", "sqm_ip", "mpsas", "frequency_hz",
              "period_counts", "period_s", "temperature_c",
              "command_bytes", "raw_response"]
    new = not path.exists() or path.stat().st_size == 0
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
           "sqm_ip": ip, **reading}
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
    return row


def test_port(ip: str, port: int, timeout: float):
    started = time.monotonic()
    with socket.create_connection((ip, port), timeout=timeout):
        elapsed = (time.monotonic() - started) * 1000
    print(f"TCP test OK: {ip}:{port} accepted a connection in {elapsed:.0f} ms")


def main():
    p = argparse.ArgumentParser(description="Read and log a Unihedron SQM-LE")
    p.add_argument("ip")
    p.add_argument("--port", type=int, default=10001)
    p.add_argument("--output", type=Path, default=Path("sqm_measurements.csv"))
    p.add_argument("--interval", type=float)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--connect-wait", type=float, default=0.25)
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"SQM-LE: {args.ip}:{args.port}")
    try:
        test_port(args.ip, args.port, args.timeout)
    except OSError as exc:
        print(f"TCP connection failed: {exc}", file=sys.stderr)
        print("Check the IP address, port, routing, firewall, and instrument power.", file=sys.stderr)
        return 2

    if args.test_only:
        return 0

    completed = 0
    target = 1 if args.interval is None else args.count
    try:
        while target == 0 or completed < target:
            start = time.monotonic()
            try:
                reading = auto_read(args.ip, args.port, args.timeout,
                                    args.connect_wait, args.verbose)
                row = save(args.output, args.ip, reading)
                completed += 1
                print(f"{row['timestamp_utc']} | {row['mpsas']:.2f} mpsas | "
                      f"{row['temperature_c']:.1f} C | saved to {args.output}")
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                print("\nThe TCP port is reachable but the SQM is silent. Close UDM or any "
                      "other SQM software, power-cycle the SQM-LE, and retry. If it remains "
                      "silent, restore the XPort defaults in UDM.", file=sys.stderr)
                if args.interval is None:
                    return 3

            if args.interval is None:
                break
            time.sleep(max(0, args.interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        print("\nStopped by user.")

    return 0 if completed else 3


if __name__ == "__main__":
    raise SystemExit(main())
