#!/usr/bin/env python3
"""Analyse SQM TDMS measurements using a calibrated reference detector.

Workflow
--------
1. Read wavelength and M-Sig/M-Dark channels from the TDMS Measurements group.
2. Treat the selected reference channel(s) as calibrated photodiodes.
3. Dark-correct every reference: I_ref = M-SigN - M-DarkN.
4. Average valid positive reference currents when several references are supplied.
5. Read PTB spectral responsivity s(lambda) from the calibration workbook.
6. Calculate optical power P(lambda) = I_ref(lambda) / s(lambda).
7. Calculate irradiance E(lambda) = P(lambda) / aperture_area.
8. Plot irradiance and each SQM signal.
9. Calculate F0 for each SQM.

Default F0 definition
---------------------
    F0 = integral(SQM_signal d_lambda) / integral(E d_lambda)

This makes F0 a band-integrated SQM-signal-to-irradiance factor. If your
laboratory defines F0 differently, change --f0-method or modify compute_f0().

Example
-------
python run_irradiance_f0.py measurement.tdms --references 4 --sqms 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from nptdms import types as tdms_types

# npTDMS workaround for unrelated ExtendedFloat channels.
tdms_types.ExtendedFloat.size = 16


def compact_name(value: object) -> str:
    """Return a lower-case alphanumeric name for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_group(tdms_file: TdmsFile, requested: str):
    target = compact_name(requested)
    for group in tdms_file.groups():
        if compact_name(group.name) == target:
            return group
    available = [group.name for group in tdms_file.groups()]
    raise ValueError(
        f"Group {requested!r} not found. Available groups: {available}"
    )


def channel_map(group) -> dict[str, str]:
    return {compact_name(channel.name): channel.name for channel in group.channels()}


def find_channel(group, names: list[str], required: bool = True) -> str | None:
    lookup = channel_map(group)
    for name in names:
        matched = lookup.get(compact_name(name))
        if matched is not None:
            return matched
    if required:
        available = [channel.name for channel in group.channels()]
        raise ValueError(
            f"Required channel not found. Tried {names}; available: {available}"
        )
    return None


def indexed_channels(group, stem: str) -> dict[int, str]:
    target = compact_name(stem)
    found: dict[int, str] = {}
    for channel in group.channels():
        match = re.fullmatch(re.escape(target) + r"(\d+)", compact_name(channel.name))
        if match:
            found[int(match.group(1))] = channel.name
    return found


def read_float(group, channel_name: str, expected_length: int | None = None) -> np.ndarray:
    values = np.asarray(group[channel_name][:], dtype=float)
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"Channel {channel_name!r} has {len(values)} values; "
            f"expected {expected_length}."
        )
    return values


def load_responsivity(calibration_file: Path, sheet_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load wavelength, responsivity and responsivity uncertainty from PTB Excel."""
    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

    raw = pd.read_excel(calibration_file, sheet_name=sheet_name, engine="openpyxl")
    if raw.shape[1] < 3:
        raise ValueError("Calibration sheet must contain at least three columns.")

    cal = raw.iloc[:, :3].copy()
    cal.columns = ["wavelength_nm", "responsivity_A_W", "u_responsivity_A_W"]
    for column in cal.columns:
        cal[column] = pd.to_numeric(cal[column], errors="coerce")
    cal = cal.dropna(subset=["wavelength_nm", "responsivity_A_W"])
    cal = cal[np.isfinite(cal["wavelength_nm"]) & np.isfinite(cal["responsivity_A_W"])]
    cal = cal[cal["responsivity_A_W"] > 0]

    # The workbook can contain further/interpolated tables to the right. Only
    # the first three columns are used. Duplicate wavelengths are consolidated.
    cal = cal.groupby("wavelength_nm", as_index=False).first().sort_values("wavelength_nm")
    if len(cal) < 2:
        raise ValueError("Calibration table contains fewer than two valid points.")

    return (
        cal["wavelength_nm"].to_numpy(float),
        cal["responsivity_A_W"].to_numpy(float),
        cal["u_responsivity_A_W"].to_numpy(float),
    )


def interpolate_without_extrapolation(
    wavelength_nm: np.ndarray,
    calibration_wavelength_nm: np.ndarray,
    calibration_values: np.ndarray,
) -> np.ndarray:
    """Linear interpolation, returning NaN outside the calibrated interval."""
    result = np.interp(
        wavelength_nm,
        calibration_wavelength_nm,
        calibration_values,
        left=np.nan,
        right=np.nan,
    )
    return result


def compute_f0(
    wavelength_nm: np.ndarray,
    sqm_signal: np.ndarray,
    irradiance_W_m2: np.ndarray,
    method: str,
) -> float:
    """Calculate an SQM factor using the selected definition."""
    valid = (
        np.isfinite(wavelength_nm)
        & np.isfinite(sqm_signal)
        & np.isfinite(irradiance_W_m2)
        & (irradiance_W_m2 > 0)
    )
    if np.count_nonzero(valid) < 2:
        return float("nan")

    wl = wavelength_nm[valid]
    signal = sqm_signal[valid]
    irradiance = irradiance_W_m2[valid]
    order = np.argsort(wl)
    wl, signal, irradiance = wl[order], signal[order], irradiance[order]

    if method == "ratio-integrals":
        denominator = np.trapz(irradiance, wl)
        return float(np.trapz(signal, wl) / denominator) if denominator != 0 else float("nan")

    if method == "mean-ratio":
        return float(np.mean(signal / irradiance))

    if method == "origin-slope":
        # Least-squares signal = F0 * irradiance, constrained through the origin.
        denominator = float(np.dot(irradiance, irradiance))
        return float(np.dot(irradiance, signal) / denominator) if denominator != 0 else float("nan")

    raise ValueError(f"Unknown F0 method: {method}")


def save_irradiance_plot(
    wavelength_nm: np.ndarray,
    irradiance_W_m2: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(wavelength_nm, irradiance_W_m2, marker="o", linewidth=1.4)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Irradiance (W/m²)")
    ax.set_title("Irradiance from calibrated reference detector")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_combined_plot(
    wavelength_nm: np.ndarray,
    irradiance_W_m2: np.ndarray,
    sqm_signals: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(wavelength_nm, irradiance_W_m2, marker="o", linewidth=1.4)
    ax1.set_ylabel("Irradiance (W/m²)")
    ax1.set_title("Reference irradiance and SQM signals")
    ax1.grid(True, alpha=0.3)

    for sqm_id, values in sqm_signals.items():
        ax2.plot(wavelength_nm, values, marker="o", linewidth=1.2, label=f"SQM {sqm_id}")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("SQM signal")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def analyse(
    tdms_path: Path,
    reference_ids: list[int],
    sqm_ids: list[int],
    output_dir: Path,
    measurement_group_name: str,
    calibration_file: Path,
    calibration_sheet: str,
    aperture_diameter_mm: float,
    f0_method: str,
) -> None:
    reference_ids = sorted(set(reference_ids))
    sqm_ids = sorted(set(sqm_ids))
    if set(reference_ids) & set(sqm_ids):
        raise ValueError("An instrument cannot be both a reference and an SQM.")
    if aperture_diameter_mm <= 0:
        raise ValueError("Aperture diameter must be positive.")
    if not tdms_path.exists():
        raise FileNotFoundError(f"TDMS file not found: {tdms_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tdms_file = TdmsFile.read(tdms_path)
    measurements = find_group(tdms_file, measurement_group_name)

    wl_channel = find_channel(measurements, ["WL /nm", "WL", "Wavelength", "Wavelength nm"])
    wavelength_nm = read_float(measurements, wl_channel)
    n = len(wavelength_nm)

    signal_channels = indexed_channels(measurements, "M-Sig")
    dark_channels = indexed_channels(measurements, "M-Dark")

    missing_sqms = [idx for idx in sqm_ids if idx not in signal_channels]
    if missing_sqms:
        raise ValueError(f"Missing M-Sig channels for SQM indices: {missing_sqms}")

    missing_references = [
        idx for idx in reference_ids
        if idx not in signal_channels or idx not in dark_channels
    ]
    if missing_references:
        raise ValueError(
            "Every reference requires M-SigN and M-DarkN. Missing indices: "
            f"{missing_references}"
        )

    sqm_signals = {
        idx: read_float(measurements, signal_channels[idx], n)
        for idx in sqm_ids
    }

    reference_net_by_id: dict[int, np.ndarray] = {}
    for idx in reference_ids:
        reference_net_by_id[idx] = (
            read_float(measurements, signal_channels[idx], n)
            - read_float(measurements, dark_channels[idx], n)
        )

    # Average only finite positive reference currents point by point.
    reference_stack = np.vstack([reference_net_by_id[idx] for idx in reference_ids])
    valid_reference_stack = np.where(
        np.isfinite(reference_stack) & (reference_stack > 0),
        reference_stack,
        np.nan,
    )
    with np.errstate(invalid="ignore"):
        reference_current_A = np.nanmean(valid_reference_stack, axis=0)

    cal_wl, cal_s, cal_u_s = load_responsivity(calibration_file, calibration_sheet)
    responsivity_A_W = interpolate_without_extrapolation(wavelength_nm, cal_wl, cal_s)
    responsivity_unc_A_W = interpolate_without_extrapolation(wavelength_nm, cal_wl, cal_u_s)

    power_W = np.full(n, np.nan)
    valid_power = (
        np.isfinite(reference_current_A)
        & np.isfinite(responsivity_A_W)
        & (responsivity_A_W > 0)
    )
    power_W[valid_power] = reference_current_A[valid_power] / responsivity_A_W[valid_power]

    aperture_area_m2 = np.pi * (aperture_diameter_mm / 1000.0 / 2.0) ** 2
    irradiance_W_m2 = power_W / aperture_area_m2

    # Responsivity contribution only. Other uncertainty terms are not included.
    relative_responsivity_unc = np.full(n, np.nan)
    valid_unc = (
        np.isfinite(responsivity_unc_A_W)
        & np.isfinite(responsivity_A_W)
        & (responsivity_A_W > 0)
    )
    relative_responsivity_unc[valid_unc] = (
        responsivity_unc_A_W[valid_unc] / responsivity_A_W[valid_unc]
    )
    power_unc_W = np.abs(power_W) * relative_responsivity_unc
    irradiance_unc_W_m2 = power_unc_W / aperture_area_m2

    f0_values = {
        idx: compute_f0(wavelength_nm, values, irradiance_W_m2, f0_method)
        for idx, values in sqm_signals.items()
    }

    # Detailed wavelength-by-wavelength results.
    result_data: dict[str, np.ndarray] = {
        "Wavelength_nm": wavelength_nm,
        "ReferenceCurrent_A": reference_current_A,
        "Responsivity_A_W": responsivity_A_W,
        "ResponsivityUncertainty_A_W": responsivity_unc_A_W,
        "OpticalPower_W": power_W,
        "OpticalPower_uW": power_W * 1e6,
        "Irradiance_W_m2": irradiance_W_m2,
        "IrradianceUncertainty_W_m2_responsivity_only": irradiance_unc_W_m2,
    }
    for idx, values in reference_net_by_id.items():
        result_data[f"ReferenceNetCurrent_{idx}_A"] = values
    for idx, values in sqm_signals.items():
        result_data[f"SQM_{idx}_Signal"] = values

    pd.DataFrame(result_data).to_csv(output_dir / "irradiance_results.csv", index=False)

    # Compact F0 summary.
    with (output_dir / "f0_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SQM", "F0", "Method"])
        for idx in sqm_ids:
            writer.writerow([idx, f0_values[idx], f0_method])

    save_irradiance_plot(
        wavelength_nm,
        irradiance_W_m2,
        output_dir / "irradiance_vs_wavelength.png",
    )
    save_combined_plot(
        wavelength_nm,
        irradiance_W_m2,
        sqm_signals,
        output_dir / "irradiance_and_sqms.png",
    )

    print(f"TDMS file: {tdms_path}")
    print(f"Calibration: {calibration_file}")
    print(f"Aperture diameter: {aperture_diameter_mm:g} mm")
    print(f"Aperture area: {aperture_area_m2:.10e} m²")
    print(f"F0 method: {f0_method}")
    for idx in sqm_ids:
        print(f"F0 SQM {idx}: {f0_values[idx]:.10e}")
    print(f"Results saved in: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute calibrated irradiance and F0 for multiple SQMs."
    )
    parser.add_argument("tdms", type=Path, help="Input TDMS file")
    parser.add_argument(
        "--references", type=int, nargs="+", default=[4],
        help="Reference detector indices; default: 4",
    )
    parser.add_argument(
        "--sqms", type=int, nargs="+", default=[1, 2, 3],
        help="SQM indices; default: 1 2 3",
    )
    parser.add_argument(
        "--group", default="Measurements", help="TDMS measurement group name",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("CALIBRATION/Kalibrierschein-73321-23-PTB.xlsx"),
        help="PTB calibration workbook",
    )
    parser.add_argument(
        "--calibration-sheet",
        default="73321 23 PTB",
        help="Calibration worksheet name",
    )
    parser.add_argument(
        "--aperture-mm", type=float, default=8.0056,
        help="Circular aperture diameter in millimetres; default: 8.0056",
    )
    parser.add_argument(
        "--f0-method",
        choices=["ratio-integrals", "mean-ratio", "origin-slope"],
        default="ratio-integrals",
        help="F0 calculation definition; default: ratio-integrals",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("sqm_results"),
        help="Output directory; default: sqm_results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        analyse(
            tdms_path=args.tdms,
            reference_ids=args.references,
            sqm_ids=args.sqms,
            output_dir=args.output,
            measurement_group_name=args.group,
            calibration_file=args.calibration,
            calibration_sheet=args.calibration_sheet,
            aperture_diameter_mm=args.aperture_mm,
            f0_method=args.f0_method,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
