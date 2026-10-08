#!/usr/bin/env python3
"""Analyse multi-instrument SQM measurements stored in a TDMS file.

Examples
--------
Three SQMs (1, 2, 3) and one reference detector (4):

    python sqm_tdms_response.py sqm0001.tdms --references 4 --sqms 1 2 3

Rules
-----
* SQMs use M-SigN directly. M-DarkN is NOT subtracted.
* References use net current = M-SigN - M-DarkN.
* Several references are allowed; their positive net currents are averaged.
* Instruments are discovered from M-Sig1 ... M-SigN in Measurements.
* The result is a relative normalized response, not absolute irradiance.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from nptdms import types as tdms_types

# npTDMS cannot decode TDMS ExtendedFloat data. Giving the storage size lets
# npTDMS skip unrelated ExtendedFloat channels while ordinary FLOAT channels
# in Measurements remain readable.
tdms_types.ExtendedFloat.size = 16


def compact_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_group(tdms_file: TdmsFile, requested: str) -> object:
    target = compact_name(requested)
    for group in tdms_file.groups():
        if compact_name(group.name) == target:
            return group
    available = [g.name for g in tdms_file.groups()]
    raise ValueError(f"Group {requested!r} not found. Available groups: {available}")


def channel_map(group: object) -> dict[str, str]:
    return {compact_name(channel.name): channel.name for channel in group.channels()}


def find_channel(group: object, names: list[str], required: bool = True) -> str | None:
    lookup = channel_map(group)
    for name in names:
        if compact_name(name) in lookup:
            return lookup[compact_name(name)]
    if required:
        raise ValueError(f"Required channel not found: {names}. Found: {[c.name for c in group.channels()]}")
    return None


def indexed_channels(group: object, stem: str) -> dict[int, str]:
    target = compact_name(stem)
    found: dict[int, str] = {}
    for channel in group.channels():
        match = re.fullmatch(re.escape(target) + r"(\d+)", compact_name(channel.name))
        if match:
            found[int(match.group(1))] = channel.name
    return found


def read_float(group: object, channel_name: str, expected_length: int | None = None) -> np.ndarray:
    values = np.asarray(group[channel_name][:], dtype=float)
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"Channel {channel_name!r} has {len(values)} values; expected {expected_length}."
        )
    return values


def normalize(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    scale = np.nanmax(np.abs(values[valid]))
    if not np.isfinite(scale) or scale == 0:
        return result
    result[valid] = values[valid] / scale
    return result


def analyse(
    tdms_path: Path,
    reference_ids: list[int],
    sqm_ids: list[int],
    output_dir: Path,
    measurement_group_name: str,
) -> None:
    reference_ids = sorted(set(reference_ids))
    sqm_ids = sorted(set(sqm_ids))

    if not tdms_path.is_file():
        raise FileNotFoundError(f"TDMS file not found: {tdms_path}")
    if not reference_ids:
        raise ValueError("Select at least one reference instrument.")
    if not sqm_ids:
        raise ValueError("Select at least one SQM instrument.")
    overlap = sorted(set(reference_ids) & set(sqm_ids))
    if overlap:
        raise ValueError(f"These instruments are both reference and SQM: {overlap}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy opening avoids decoding unrelated unsupported channels.
    with TdmsFile.open(tdms_path) as tdms_file:
        measurements = find_group(tdms_file, measurement_group_name)
        wl_name = find_channel(measurements, ["WL /nm", "WL/nm", "Wavelength"])
        timestamp_name = find_channel(measurements, ["TimeStamp", "Timestamp"], required=False)

        mean_signal = indexed_channels(measurements, "M-Sig")
        mean_dark = indexed_channels(measurements, "M-Dark")
        signal_std = indexed_channels(measurements, "S-Sig")
        signal_range = indexed_channels(measurements, "Range-Sig")
        temperature = indexed_channels(measurements, "T")
        detected = sorted(mean_signal)

        missing_refs = [i for i in reference_ids if i not in mean_signal]
        missing_sqms = [i for i in sqm_ids if i not in mean_signal]
        if missing_refs or missing_sqms:
            raise ValueError(
                f"Missing reference IDs {missing_refs}; missing SQM IDs {missing_sqms}; "
                f"detected instruments: {detected}"
            )

        wavelength = read_float(measurements, wl_name)
        count = len(wavelength)

        timestamps = None
        if timestamp_name:
            try:
                timestamps = np.asarray(measurements[timestamp_name][:])
                if len(timestamps) != count:
                    timestamps = None
            except Exception:
                timestamps = None

        data: dict[str, np.ndarray] = {"Wavelength_nm": wavelength}
        reference_nets: list[np.ndarray] = []

        # Dark correction is performed only for explicitly selected references.
        for idx in reference_ids:
            signal = read_float(measurements, mean_signal[idx], count)
            if idx in mean_dark:
                dark = read_float(measurements, mean_dark[idx], count)
            else:
                dark = np.zeros(count, dtype=float)
                print(f"WARNING: M-Dark{idx} missing; zero dark current used.", file=sys.stderr)
            net = signal - dark
            reference_nets.append(net)
            data[f"Reference_{idx}_signal_A"] = signal
            data[f"Reference_{idx}_dark_A"] = dark
            data[f"Reference_{idx}_net_A"] = net
            if idx in signal_std:
                data[f"Reference_{idx}_std_A"] = read_float(measurements, signal_std[idx], count)
            if idx in signal_range:
                data[f"Reference_{idx}_range_A"] = read_float(measurements, signal_range[idx], count)
            if idx in temperature:
                data[f"Reference_{idx}_temperature"] = read_float(measurements, temperature[idx], count)

        net_stack = np.vstack(reference_nets)
        positive_stack = np.where(np.isfinite(net_stack) & (net_stack > 0), net_stack, np.nan)
        with np.errstate(all="ignore"):
            combined_reference = np.nanmean(positive_stack, axis=0)
        data["Reference_combined_net_A"] = combined_reference

        # SQM M-Sig is used directly. No SQM dark subtraction occurs here.
        for idx in sqm_ids:
            sqm_reading = read_float(measurements, mean_signal[idx], count)
            sqm_linear = np.power(10.0, -0.4 * sqm_reading)
            ratio = np.full(count, np.nan, dtype=float)
            valid = (
                np.isfinite(sqm_linear)
                & np.isfinite(combined_reference)
                & (combined_reference > 0)
            )
            ratio[valid] = sqm_linear[valid] / combined_reference[valid]
            data[f"SQM_{idx}_reading_mag_arcsec2"] = sqm_reading
            data[f"SQM_{idx}_linear_relative"] = sqm_linear
            data[f"SQM_{idx}_relative_response"] = normalize(ratio)
            if idx in signal_std:
                data[f"SQM_{idx}_reading_std"] = read_float(measurements, signal_std[idx], count)
            if idx in temperature:
                data[f"SQM_{idx}_temperature"] = read_float(measurements, temperature[idx], count)

    order = np.argsort(np.where(np.isfinite(wavelength), wavelength, np.inf))
    columns = list(data)
    csv_path = output_dir / "sqm_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = (["Timestamp"] if timestamps is not None else []) + columns
        writer.writerow(header)
        for row_index in order:
            if not np.isfinite(wavelength[row_index]):
                continue
            row = []
            if timestamps is not None:
                row.append(str(timestamps[row_index]))
            row.extend(data[name][row_index] for name in columns)
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in sqm_ids:
        ax.plot(wavelength[order], data[f"SQM_{idx}_reading_mag_arcsec2"][order], "o-", label=f"SQM {idx}")
    ax.set(xlabel="Wavelength [nm]", ylabel="SQM reading [mag/arcsec^2]", title="Raw SQM spectral measurements")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sqm_raw_readings.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in reference_ids:
        ax.plot(wavelength[order], data[f"Reference_{idx}_net_A"][order], "o-", label=f"Reference {idx}")
    ax.set(xlabel="Wavelength [nm]", ylabel="Net reference current [A]", title="Reference current: M-Sig minus M-Dark")
    ax.grid(True, alpha=0.35)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reference_net_current.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in sqm_ids:
        ax.plot(wavelength[order], data[f"SQM_{idx}_relative_response"][order], "o-", label=f"SQM {idx}")
    ax.set(xlabel="Wavelength [nm]", ylabel="Normalized relative response", title="SQM relative spectral response")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sqm_relative_response.png", dpi=300)
    plt.close(fig)

    print(f"Input: {tdms_path.resolve()}")
    print(f"Detected instruments: {detected}")
    print(f"References, dark-corrected: {reference_ids}")
    print(f"SQMs, no dark correction: {sqm_ids}")
    print(f"Results: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse multi-instrument SQM TDMS measurements.")
    parser.add_argument("tdms", type=Path, help="Input TDMS file")
    parser.add_argument("--references", type=int, nargs="+", required=True, help="Reference indices, e.g. 4 or 4 5")
    parser.add_argument("--sqms", type=int, nargs="+", required=True, help="SQM indices, e.g. 1 2 3")
    parser.add_argument("--group", default="Measurements", help="TDMS measurement group name")
    parser.add_argument("--output", type=Path, default=Path("sqm_results"), help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        analyse(args.tdms, args.references, args.sqms, args.output, args.group)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
