#!/usr/bin/env python3
"""Characterize SQMs from a TDMS scan and a calibrated reference detector.

Reference instruments:
    I_photo(lambda) = M-SigN - M-DarkN
SQM instruments:
    m(lambda) = M-SigN, without SQM dark subtraction

The reference calibration must provide wavelength [nm] and responsivity [A/W].
The program calculates optical power, irradiance, wavelength-dependent equivalent
zero points, normalized spectral response, peak wavelength, effective wavelength,
and FWHM.

Example:
python run6_multi_tdms.py visible.tdms infrared.tdms --references 4 --sqms 1 2 3 \
    --calibration responsivity.xlsx --reference-area-mm2 50.3
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

tdms_types.ExtendedFloat.size = 16
INVALID_SQM_VALUES = (65535.0,)


def compact(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_group(tdms: TdmsFile, requested: str):
    key = compact(requested)
    for group in tdms.groups():
        if compact(group.name) == key:
            return group
    raise ValueError(f"Group {requested!r} not found; available: {[g.name for g in tdms.groups()]}")


def indexed_channels(group, stem: str) -> dict[int, str]:
    key = compact(stem)
    answer = {}
    for channel in group.channels():
        match = re.fullmatch(re.escape(key) + r"(\d+)", compact(channel.name))
        if match:
            answer[int(match.group(1))] = channel.name
    return answer


def find_channel(group, candidates: list[str], required: bool = True) -> str | None:
    available = {compact(c.name): c.name for c in group.channels()}
    for candidate in candidates:
        if compact(candidate) in available:
            return available[compact(candidate)]
    if required:
        raise ValueError(f"Missing channel {candidates}; available: {[c.name for c in group.channels()]}")
    return None


def values(group, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(group[name][:], dtype=float)
    if length is not None and len(result) != length:
        raise ValueError(f"Channel {name!r} has {len(result)} values, expected {length}")
    return result


def choose_column(df: pd.DataFrame, requested: str | None, aliases: list[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Calibration column {requested!r} not found: {list(df.columns)}")
        return requested
    lookup = {compact(c): c for c in df.columns}
    for alias in aliases:
        if compact(alias) in lookup:
            return lookup[compact(alias)]
    raise ValueError(f"Could not identify calibration column from {list(df.columns)}")


def load_calibration(path: Path, sheet: str | None, wavelength_col: str | None,
                     responsivity_col: str | None, uncertainty_col: str | None):
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(path, sheet_name=sheet or 0, engine="openpyxl")
    else:
        df = pd.read_csv(path)
    # PTB workbooks commonly use Unicode headings such as
    # "lambda / nm" (Greek lambda) and "s(lambda) / (A/W)".
    # Prefer the user's explicit names, then exact PTB-style columns, then
    # generic aliases. This changes only calibration-column identification.
    if wavelength_col:
        wcol = choose_column(df, wavelength_col, [])
    else:
        wcol = next(
            (c for c in df.columns if str(c).strip() in {"lambda / nm", "Lambda / nm", "λ / nm"}),
            None,
        )
        if wcol is None:
            wcol = choose_column(df, None, ["wavelength", "wavelength nm", "lambda", "nm"])

    if responsivity_col:
        rcol = choose_column(df, responsivity_col, [])
    else:
        # Select the first base responsivity column, not interpolation copies
        # carrying suffixes such as ".1" or ".2".
        rcol = next(
            (
                c for c in df.columns
                if re.match(r"^s\s*\(.*\)\s*/\s*\(A/W\)$", str(c).strip(), re.IGNORECASE)
                and not re.search(r"\.\d+$", str(c).strip())
            ),
            None,
        )
        if rcol is None:
            rcol = choose_column(df, None, ["responsivity", "responsivity a/w", "spectral responsivity", "a/w"])

    ucol = None
    if uncertainty_col:
        ucol = choose_column(df, uncertainty_col, [])
    else:
        try:
            ucol = choose_column(df, None, ["uncertainty", "relative uncertainty", "uncertainty percent", "u percent"])
        except ValueError:
            pass
    cal = pd.DataFrame({"wl": pd.to_numeric(df[wcol], errors="coerce"),
                        "resp": pd.to_numeric(df[rcol], errors="coerce")})
    if ucol:
        cal["u"] = pd.to_numeric(df[ucol], errors="coerce")
    cal = cal.dropna(subset=["wl", "resp"]).sort_values("wl").drop_duplicates("wl")
    if len(cal) < 2 or (cal["resp"] <= 0).any():
        raise ValueError("Calibration needs at least two positive responsivity points")
    return cal


def interpolate(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp, left=np.nan, right=np.nan)


def normalize(y: np.ndarray) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    valid = np.isfinite(y) & (y >= 0)
    if valid.any() and np.nanmax(y[valid]) > 0:
        out[valid] = y[valid] / np.nanmax(y[valid])
    return out


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def effective_wavelength(wl: np.ndarray, response: np.ndarray) -> float:
    valid = np.isfinite(wl) & np.isfinite(response) & (response >= 0)
    if valid.sum() < 2:
        return np.nan
    x, y = wl[valid], response[valid]
    area = trapz(y, x)
    return trapz(x * y, x) / area if area > 0 else np.nan


def crossing(x1, y1, x2, y2, level=0.5):
    if y2 == y1:
        return (x1 + x2) / 2
    return x1 + (level - y1) * (x2 - x1) / (y2 - y1)


def fwhm(wl: np.ndarray, response: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(wl) & np.isfinite(response)
    x, y = wl[valid], response[valid]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    order = np.argsort(x); x, y = x[order], y[order]
    peak = int(np.nanargmax(y))
    left = right = np.nan
    for i in range(peak - 1, -1, -1):
        if (y[i] - 0.5) * (y[i + 1] - 0.5) <= 0:
            left = crossing(x[i], y[i], x[i + 1], y[i + 1]); break
    for i in range(peak, len(x) - 1):
        if (y[i] - 0.5) * (y[i + 1] - 0.5) <= 0:
            right = crossing(x[i], y[i], x[i + 1], y[i + 1]); break
    return left, right, right - left if np.isfinite(left) and np.isfinite(right) else np.nan




def load_cie_function(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column wavelength[nm], spectral-efficiency CSV without header."""
    frame = pd.read_csv(path, header=None, comment="#")
    if frame.shape[1] < 2:
        raise ValueError(f"CIE file {path} must contain at least two columns")
    wl = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    efficiency = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(wl) & np.isfinite(efficiency) & (efficiency >= 0)
    wl, efficiency = wl[valid], efficiency[valid]
    order = np.argsort(wl)
    if len(wl) < 2:
        raise ValueError(f"CIE file {path} has insufficient valid data")
    return wl[order], efficiency[order]

def read_tdms_measurements(tdms_path: Path, group_name: str,
                           refs: list[int], sqms: list[int]):
    """Read one TDMS file and return its wavelength, reference, and SQM arrays."""
    with TdmsFile.open(tdms_path) as tdms:
        group = find_group(tdms, group_name)
        wl_name = find_channel(group, ["WL /nm", "WL/nm", "Wavelength"])
        msig = indexed_channels(group, "M-Sig")
        mdark = indexed_channels(group, "M-Dark")
        detected = sorted(msig)
        missing = [i for i in refs + sqms if i not in msig]
        if missing:
            raise ValueError(
                f"In {tdms_path}: missing instruments {missing}; detected {detected}"
            )

        wavelength = values(group, wl_name)
        n = len(wavelength)
        ref_signals, ref_darks, ref_nets = {}, {}, {}
        for idx in refs:
            signal = values(group, msig[idx], n)
            dark = values(group, mdark[idx], n) if idx in mdark else np.zeros(n)
            net = signal - dark
            ref_signals[idx] = signal
            ref_darks[idx] = dark
            ref_nets[idx] = net

        sqm_readings = {}
        for idx in sqms:
            m = values(group, msig[idx], n)
            invalid = ~np.isfinite(m)
            for marker in INVALID_SQM_VALUES:
                invalid |= np.isclose(m, marker)
            sqm_readings[idx] = np.where(invalid, np.nan, m)

    return {
        "source_file": np.full(n, tdms_path.name, dtype=object),
        "wavelength": wavelength,
        "ref_signals": ref_signals,
        "ref_darks": ref_darks,
        "ref_nets": ref_nets,
        "sqm_readings": sqm_readings,
        "detected": detected,
    }


def analyse(args):
    refs, sqms = sorted(set(args.references)), sorted(set(args.sqms))
    if set(refs) & set(sqms):
        raise ValueError("An instrument cannot be both reference and SQM")
    if args.reference_area_mm2 <= 0:
        raise ValueError("Reference area must be positive")
    if len(args.tdms) < 1:
        raise ValueError("At least one TDMS file is required")

    cal = load_calibration(args.calibration, args.calibration_sheet,
                           args.wavelength_column, args.responsivity_column,
                           args.uncertainty_column)

    scans = [read_tdms_measurements(path, args.group, refs, sqms) for path in args.tdms]
    wavelength = np.concatenate([scan["wavelength"] for scan in scans])
    source_file = np.concatenate([scan["source_file"] for scan in scans])
    detected_by_file = {str(path): scan["detected"] for path, scan in zip(args.tdms, scans)}

    result: dict[str, np.ndarray] = {
        "Source_TDMS": source_file,
        "Wavelength_nm": wavelength,
    }

    ref_nets_combined = []
    for idx in refs:
        signal = np.concatenate([scan["ref_signals"][idx] for scan in scans])
        dark = np.concatenate([scan["ref_darks"][idx] for scan in scans])
        net = np.concatenate([scan["ref_nets"][idx] for scan in scans])
        ref_nets_combined.append(np.where(net > 0, net, np.nan))
        result[f"Reference_{idx}_signal_A"] = signal
        result[f"Reference_{idx}_dark_A"] = dark
        result[f"Reference_{idx}_net_A"] = net

    with np.errstate(all="ignore"):
        current = np.nanmean(np.vstack(ref_nets_combined), axis=0)
    result["Reference_combined_net_A"] = current

    sqm_readings = {}
    for idx in sqms:
        m = np.concatenate([scan["sqm_readings"][idx] for scan in scans])
        sqm_readings[idx] = m
        result[f"SQM_{idx}_mag_arcsec2"] = m

    responsivity = interpolate(wavelength, cal["wl"].to_numpy(), cal["resp"].to_numpy())
    area_m2 = args.reference_area_mm2 * 1e-6
    optical_power = current / responsivity
    irradiance = optical_power / area_m2
    result["Reference_responsivity_A_W"] = responsivity
    result["Optical_power_W"] = optical_power
    result["Irradiance_W_m2"] = irradiance

    # For each narrow monochromator band, illuminance is obtained from the
    # band irradiance and the CIE spectral luminous-efficiency function.
    cie_p_wl, cie_p_v = load_cie_function(args.cie_photopic)
    v_photopic = interpolate(wavelength, cie_p_wl, cie_p_v)
    photopic_illuminance = 683.0 * irradiance * v_photopic
    result["CIE_photopic_V_lambda"] = v_photopic
    result["Photopic_illuminance_lx"] = photopic_illuminance

    scotopic_illuminance = None
    v_scotopic = None
    if args.cie_scotopic is not None:
        cie_s_wl, cie_s_v = load_cie_function(args.cie_scotopic)
        v_scotopic = interpolate(wavelength, cie_s_wl, cie_s_v)
        scotopic_illuminance = 1700.0 * irradiance * v_scotopic
        result["CIE_scotopic_Vprime_lambda"] = v_scotopic
        result["Scotopic_illuminance_lx"] = scotopic_illuminance

    summary_rows = []
    for idx, m in sqm_readings.items():
        linear = np.power(10.0, -0.4 * m)
        response_per_irradiance = linear / irradiance
        relative = normalize(response_per_irradiance)
        e0 = irradiance * np.power(10.0, 0.4 * m)
        result[f"SQM_{idx}_linear_relative"] = linear
        result[f"SQM_{idx}_response_per_irradiance"] = response_per_irradiance
        result[f"SQM_{idx}_relative_spectral_response"] = relative
        result[f"SQM_{idx}_E0_W_m2"] = e0
        # Wavelength-dependent illuminance-equivalent zero points.
        # Values become undefined where the corresponding CIE efficiency is
        # unavailable; very small efficiencies create numerically fragile values.
        result[f"SQM_{idx}_photopic_Ev0_lx"] = photopic_illuminance * np.power(10.0, 0.4 * m)
        if scotopic_illuminance is not None:
            result[f"SQM_{idx}_scotopic_Ev0_lx"] = scotopic_illuminance * np.power(10.0, 0.4 * m)
        valid = np.isfinite(relative)
        peak_wl = wavelength[valid][np.nanargmax(relative[valid])] if valid.any() else np.nan
        left, right, width = fwhm(wavelength, relative)
        summary_rows.append({"SQM": idx, "Peak_wavelength_nm": peak_wl,
                             "Effective_wavelength_nm": effective_wavelength(wavelength, relative),
                             "FWHM_lower_nm": left, "FWHM_upper_nm": right,
                             "FWHM_nm": width, "Valid_points": int(valid.sum()),
                             "Median_E0_W_m2": float(np.nanmedian(e0))})

    order = np.argsort(np.where(np.isfinite(wavelength), wavelength, np.inf))
    args.output.mkdir(parents=True, exist_ok=True)
    combined_frame = pd.DataFrame({k: v[order] for k, v in result.items()})
    combined_frame.to_csv(args.output / "sqm_characterization.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output / "sqm_summary.csv", index=False)

    # ==========================================================
    # FIGURE: RAW MEASUREMENTS FROM ALL DEVICES
    # ==========================================================

    fig, (ax_sqm, ax_ref) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Upper panel: raw SQM readings
    for idx in sqms:
        sqm_values = result[f"SQM_{idx}_mag_arcsec2"][order]

        valid_sqm = (
            np.isfinite(wavelength[order])
            & np.isfinite(sqm_values)
        )

        ax_sqm.plot(
            wavelength[order][valid_sqm],
            sqm_values[valid_sqm],
            "o-",
            linewidth=1.5,
            markersize=4,
            label=f"SQM {idx}"
        )

    ax_sqm.set_ylabel("SQM reading [mag/arcsec²]")
    ax_sqm.set_title("Raw measurements from all devices")
    ax_sqm.grid(True, alpha=0.35)
    ax_sqm.legend()

    # Optional:
    # Lower magnitude means more light. Uncomment the following line
    # if you want brighter measurements to appear higher on the graph.
    #
    # ax_sqm.invert_yaxis()

    # Lower panel: dark-corrected reference current
    for idx in refs:
        reference_values = result[f"Reference_{idx}_net_A"][order]

        valid_reference = (
            np.isfinite(wavelength[order])
            & np.isfinite(reference_values)
            & (reference_values > 0)
        )

        ax_ref.plot(
            wavelength[order][valid_reference],
            reference_values[valid_reference],
            "o-",
            linewidth=1.5,
            markersize=4,
            label=f"Reference {idx}"
        )

    ax_ref.set_xlabel("Wavelength [nm]")
    ax_ref.set_ylabel("Net current [A]")
    ax_ref.grid(True, alpha=0.35)
    ax_ref.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )
    ax_ref.legend()

    fig.tight_layout()

    fig.savefig(
        args.output / "all_devices_raw_measurements.png",
        dpi=300
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in sqms:
        ax.plot(wavelength[order], result[f"SQM_{idx}_relative_spectral_response"][order], "o-", label=f"SQM {idx}")
    ax.set(xlabel="Wavelength [nm]", ylabel="Normalized response", title="SQM relative spectral response")
    ax.grid(True, alpha=.35); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "sqm_relative_spectral_response.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelength[order], irradiance[order], "o-")
    ax.set(xlabel="Wavelength [nm]", ylabel="Irradiance [W/m2]", title="Irradiance from reference detector")
    ax.grid(True, alpha=.35); fig.tight_layout()
    fig.savefig(args.output / "irradiance.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in sqms:
        ax.plot(wavelength[order], result[f"SQM_{idx}_E0_W_m2"][order], "o-", label=f"SQM {idx}")

    ax.set_yscale("log")
    ax.set(xlabel="Wavelength [nm]", ylabel="Equivalent E0 [W/m2]", title="SQM wavelength-dependent irradiance-equivalent zero point")
    ax.grid(True, alpha=.35); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "sqm_E0.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelength[order], photopic_illuminance[order], "o-", label="Photopic")
    if scotopic_illuminance is not None:
        ax.plot(wavelength[order], scotopic_illuminance[order], "o-", label="Scotopic")
    ax.set(xlabel="Wavelength [nm]", ylabel="Illuminance [lx]",
           title="Illuminance from reference-detector irradiance")
    ax.grid(True, alpha=.35); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "illuminance.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in sqms:
        y = result[f"SQM_{idx}_photopic_Ev0_lx"][order]
        good = np.isfinite(y) & (y > 0)
        ax.plot(wavelength[order][good], y[good], "o-", label=f"SQM {idx}")
    ax.set_yscale("log")
    ax.set(xlabel="Wavelength [nm]", ylabel="Photopic illuminance-equivalent Ev0 [lx]",
           title="SQM wavelength-dependent photopic illuminance-equivalent zero point")
    ax.grid(True, which="both", alpha=.35); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "sqm_photopic_Ev0.png", dpi=300); plt.close(fig)

    if scotopic_illuminance is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx in sqms:
            y = result[f"SQM_{idx}_scotopic_Ev0_lx"][order]
            good = np.isfinite(y) & (y > 0)
            ax.plot(wavelength[order][good], y[good], "o-", label=f"SQM {idx}")
        ax.set_yscale("log")
        ax.set(xlabel="Wavelength [nm]", ylabel="Scotopic illuminance-equivalent Ev0 [lx]",
               title="SQM wavelength-dependent scotopic illuminance-equivalent zero point")
        ax.grid(True, which="both", alpha=.35); ax.legend(); fig.tight_layout()
        fig.savefig(args.output / "sqm_scotopic_Ev0.png", dpi=300); plt.close(fig)

    print("Input TDMS files:")
    for path in args.tdms:
        print(f"  {path} | detected instruments: {detected_by_file[str(path)]}")
    print(f"Combined wavelength points: {len(wavelength)}")
    print(f"Combined wavelength range: {np.nanmin(wavelength):g} to {np.nanmax(wavelength):g} nm")
    print(f"References with dark correction: {refs}")
    print(f"SQMs without dark correction: {sqms}")
    print(f"Results written to: {args.output.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Characterize SQMs using TDMS and a calibrated photodetector")
    p.add_argument(
        "tdms", type=Path, nargs="+",
        help="One or more input TDMS files; measurements are concatenated and sorted by wavelength",
    )
    p.add_argument("--references", type=int, nargs="+", default=[4])
    p.add_argument("--sqms", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--group", default="Measurements")
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--calibration-sheet", default=None)
    p.add_argument("--wavelength-column", default=None)
    p.add_argument("--responsivity-column", default=None)
    p.add_argument("--uncertainty-column", default=None)
    p.add_argument("--reference-area-mm2", type=float, required=True,
                   help="Illuminated active area of reference detector in mm2")
    p.add_argument("--cie-photopic", type=Path, required=True,
                   help="Two-column CIE photopic V(lambda) CSV")
    p.add_argument("--cie-scotopic", type=Path, default=None,
                   help="Optional two-column CIE scotopic V-prime(lambda) CSV")
    p.add_argument("--output", type=Path, default=Path("sqm_results"))
    return p.parse_args()


def main():
    try:
        analyse(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
