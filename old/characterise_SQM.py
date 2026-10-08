#!/usr/bin/env python3
"""Characterise an SQM against a BTS256-EF Excel reference export.

Expected inputs
---------------
DATA/sqm_measurements_SQM3_intensityX.csv
ref_bts_with_minisphere.xls

The BTS export is matched to SQM files through its ``comment`` row. For each
SQM file, the final three valid MPSAS readings are averaged, or all available
values when fewer than three exist. BTS spectral irradiance is converted to
spectral radiance using L_lambda = E_lambda / pi, appropriate for a uniform
Lambertian sphere exit.

Outputs are written to RESULTS_SQM_CHARACTERISATION by default.
"""

from __future__ import annotations
import os
import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INTENSITY_RE = re.compile(r"intensity\s*[_-]?\s*(\d+)", re.IGNORECASE)


def norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def intensity_number(text: object) -> int | None:
    match = INTENSITY_RE.search(str(text))
    return int(match.group(1)) if match else None


def read_bts_xls(path: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray], np.ndarray]:
    """Return BTS metadata, irradiance spectra by intensity, and wavelengths."""
    try:
        raw = pd.read_excel(path, header=None, engine="calamine")
    except ImportError as exc:
        raise ImportError(
            "Reading legacy .xls files requires python-calamine. Install with:\n"
            "    python -m pip install python-calamine"
        ) from exc

    labels = raw.iloc[:, 0].map(norm)

    def find_row(label: str, last: bool = False) -> int:
        hits = raw.index[labels.eq(norm(label))].tolist()
        if not hits:
            raise ValueError(f"BTS row not found: {label!r}")
        return hits[-1] if last else hits[0]

    comment_row = find_row("comment")
    lux_row = find_row("integral sensor value")
    cct_row = find_row("CCT") if labels.eq("cct").any() else None
    radiometric_row = (
        find_row("radiometric intensity")
        if labels.eq("radiometric intensity").any()
        else None
    )
    wavelength_header = find_row("wavelength /nm", last=True)

    records = []
    column_for_intensity: dict[int, int] = {}
    for col in range(1, raw.shape[1]):
        comment = raw.iat[comment_row, col]
        number = intensity_number(comment)
        if number is None:
            continue
        if number in column_for_intensity:
            raise ValueError(f"Duplicate BTS comment for intensity {number}")
        column_for_intensity[number] = col
        records.append(
            {
                "intensity": number,
                "bts_comment": str(comment).strip(),
                "illuminance_lx": pd.to_numeric(raw.iat[lux_row, col], errors="coerce"),
                "cct_K": (
                    pd.to_numeric(raw.iat[cct_row, col], errors="coerce")
                    if cct_row is not None
                    else np.nan
                ),
                "bts_radiometric_irradiance_W_m2": (
                    pd.to_numeric(raw.iat[radiometric_row, col], errors="coerce")
                    if radiometric_row is not None
                    else np.nan
                ),
                "bts_column_1based": col,
            }
        )

    if not records:
        raise ValueError("No comments matching 'intensityX' were found in the BTS file")

    spectral_block = raw.iloc[wavelength_header + 1 :].copy()
    wavelengths = pd.to_numeric(spectral_block.iloc[:, 0], errors="coerce")
    valid_wl = wavelengths.between(100, 3000)
    wavelengths = wavelengths.loc[valid_wl].to_numpy(float)
    if wavelengths.size < 2:
        raise ValueError("No BTS wavelength table found below 'wavelength /nm'")

    spectra: dict[int, np.ndarray] = {}
    for number, col in column_for_intensity.items():
        values = pd.to_numeric(spectral_block.iloc[:, col], errors="coerce")
        spectrum = values.loc[valid_wl].to_numpy(float)
        spectra[number] = spectrum

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    spectra = {key: values[order] for key, values in spectra.items()}
    metadata = pd.DataFrame(records).sort_values("intensity").reset_index(drop=True)
    return metadata, spectra, wavelengths


def read_sqm_files(data_dir: Path, pattern: str) -> pd.DataFrame:
    records = []
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No SQM CSV files match {data_dir / pattern}")

    for path in files:
        number = intensity_number(path.stem)
        if number is None:
            print(f"Warning: ignoring file without intensity number: {path.name}")
            continue
        frame = pd.read_csv(path)
        if "mpsas" not in frame.columns:
            print(f"Warning: ignoring {path.name}; missing 'mpsas' column")
            continue
        values = pd.to_numeric(frame["mpsas"], errors="coerce").dropna()
        if values.empty:
            print(f"Warning: ignoring {path.name}; no valid MPSAS values")
            continue
        tail = values.tail(3)
        records.append(
            {
                "intensity": number,
                "sqm_file": str(path),
                "sqm_n_total": int(values.size),
                "sqm_n_averaged": int(tail.size),
                "sqm_mpsas": float(tail.mean()),
                "sqm_mpsas_std_last": float(tail.std(ddof=1)) if tail.size > 1 else np.nan,
                "sqm_last_values": ";".join(f"{value:.8g}" for value in tail),
                "sqm_temperature_C": (
                    float(pd.to_numeric(frame["temperature_c"], errors="coerce").tail(3).mean())
                    if "temperature_c" in frame.columns
                    else np.nan
                ),
            }
        )

    if not records:
        raise ValueError("No usable SQM CSV files were found")
    return pd.DataFrame(records).sort_values("intensity").reset_index(drop=True)


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    return float(np.trapezoid(y[finite], x[finite]))


def fit_fixed_magnitude_model(table: pd.DataFrame, predictor: str) -> dict[str, object]:
    """Fit m = a - 2.5 log10(L) and derive L0.

    This is equivalent to m = -2.5 log10(L / L0), with
    L0 = 10**(a / 2.5). L0 has the same unit as the predictor.
    """
    valid = (
        np.isfinite(table[predictor])
        & (table[predictor] > 0)
        & np.isfinite(table.sqm_mpsas)
    )
    radiance = table.loc[valid, predictor].to_numpy(float)
    magnitude = table.loc[valid, "sqm_mpsas"].to_numpy(float)
    if len(radiance) < 2:
        raise ValueError(
            f"At least two matched levels are required to fit against {predictor}"
        )

    log_radiance = np.log10(radiance)
    individual_intercepts = magnitude + 2.5 * log_radiance
    intercept = float(np.mean(individual_intercepts))
    predicted = intercept - 2.5 * log_radiance
    residuals = magnitude - predicted

    n = len(radiance)
    residual_std = float(np.std(residuals, ddof=1)) if n > 1 else np.nan
    intercept_standard_uncertainty = residual_std / np.sqrt(n) if n > 1 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((magnitude - magnitude.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # From a = 2.5 log10(L0).
    reference_radiance_L0 = float(10 ** (intercept / 2.5))

    # Propagate the standard uncertainty of a to L0.
    if np.isfinite(intercept_standard_uncertainty):
        relative_u_L0 = float(
            np.log(10.0) / 2.5 * intercept_standard_uncertainty
        )
        standard_u_L0 = reference_radiance_L0 * relative_u_L0
    else:
        relative_u_L0 = np.nan
        standard_u_L0 = np.nan

    return {
        "predictor": predictor,
        "n": int(n),
        "model": "m = a - 2.5 log10(Le)",
        "equivalent_model": "m = -2.5 log10(Le / L0)",
        "fixed_slope": -2.5,
        "intercept_a": intercept,
        "reference_radiance_L0": reference_radiance_L0,
        "residual_standard_deviation_magnitude": residual_std,
        "standard_uncertainty_intercept_a": float(intercept_standard_uncertainty),
        "standard_uncertainty_L0": float(standard_u_L0),
        "relative_standard_uncertainty_L0": float(relative_u_L0),
        "rmse_magnitude": rmse,
        "r2_fixed_slope": float(r2),
    }

def make_figures(table: pd.DataFrame, spectra: dict[int, np.ndarray], wavelengths: np.ndarray,
                 output: Path, model: dict[str, object], predictor: str) -> None:
    # 1. Absolute spectral radiance and normalized spectral shape.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for _, row in table.iterrows():
        number = int(row.intensity)
        if number not in spectra:
            continue
        radiance = spectra[number] / math.pi
        axes[0].plot(wavelengths, radiance, label=f"Intensity {number}")
        area = trapz(radiance, wavelengths)
        if np.isfinite(area) and area > 0:
            axes[1].plot(wavelengths, radiance / area, label=f"Intensity {number}")
    axes[0].set(xlabel="Wavelength (nm)", ylabel="Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)",
                title="BTS reference spectral radiance")
    axes[1].set(xlabel="Wavelength (nm)", ylabel="Area-normalized spectral radiance (nm$^{-1}$)",
                title="Spectral-shape change with intensity")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "01_reference_radiance_spectra.png", dpi=220)
    plt.close(fig)

    # 2. Magnitude response and free/fixed-slope fits.
    valid = np.isfinite(table[predictor]) & (table[predictor] > 0) & np.isfinite(table.sqm_mpsas)
    q = table.loc[valid, predictor].to_numpy(float)
    m = table.loc[valid, "sqm_mpsas"].to_numpy(float)
    order = np.argsort(q)
    qline = np.geomspace(q.min(), q.max(), 300)
    pred_fixed = model["intercept_a"] - 2.5 * np.log10(qline)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(q, m, s=55, color="black")
    ax.plot(
        qline,
        pred_fixed,
        label=(
            rf"$m=-2.5\log_{{10}}\left(\frac{{L_e}}{{L_0}}\right),\quad "
            rf"L_0={model['reference_radiance_L0']:.4e}\ "
            rf"\mathrm{{W\,m^{{-2}}\,sr^{{-1}}}}$"
        ),
    )
    for _, row in table.loc[valid].iterrows():
        ax.annotate(str(int(row.intensity)), (row[predictor], row.sqm_mpsas), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set(xlabel="Integrated radiance, 360-830 nm (W m$^{-2}$ sr$^{-1}$)",
           ylabel="SQM magnitude (mag arcsec$^{-2}$)", title="SQM magnitude response")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "02_sqm_magnitude_response.png", dpi=220)
    plt.close(fig)

    # 3. Residuals and relative linear signal response.
    logq = np.log10(q)
    pred = model["intercept_a"] - 2.5 * logq
    residual = m - pred
    sqm_signal = 10 ** (-0.4 * m)
    scale = float(np.sum(q * sqm_signal) / np.sum(sqm_signal**2))
    q_from_sqm = scale * sqm_signal
    ratio = q_from_sqm / q

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].scatter(q, residual, s=55)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Integrated radiance (W m$^{-2}$ sr$^{-1}$)",
                ylabel="Magnitude residual: measured - fitted",
                title="Residuals of the fixed -2.5 slope fit")
    axes[1].axhline(1, color="black", linewidth=1)
    axes[1].scatter(q, ratio, s=55)
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Integrated radiance (W m$^{-2}$ sr$^{-1}$)",
                ylabel="Scaled SQM relative signal / reference radiance",
                title="Linearity in the SQM's linearized signal domain")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "03_residuals_and_linearity.png", dpi=220)
    plt.close(fig)

    # 4. Lux comparison, useful but secondary to radiance.
    fig, ax = plt.subplots(figsize=(8, 6))
    good = np.isfinite(table.illuminance_lx) & (table.illuminance_lx > 0)
    ax.scatter(table.loc[good, "illuminance_lx"], table.loc[good, "sqm_mpsas"], s=55)
    for _, row in table.loc[good].iterrows():
        ax.annotate(str(int(row.intensity)), (row.illuminance_lx, row.sqm_mpsas),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set(xlabel="BTS integral sensor illuminance (lx)",
           ylabel="SQM magnitude (mag arcsec$^{-2}$)", title="SQM response versus illuminance")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "04_sqm_vs_illuminance.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Characterise SQM magnitude response against BTS spectra")
    parser.add_argument("--bts", type=Path, default=os.path.join("REFERENCES","BTS","ref_bts_with_minisphere.xls"),
                        help="Path to BTS Excel reference export")
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"))
    parser.add_argument("--pattern", default="sqm_measurements_*_intensity*.csv")
    parser.add_argument(
        "--intensities",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Intensity numbers to include in the characterisation, for example "
            "--intensities 1 2 3 4 5. By default, all matched intensities are used."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("RESULTS_SQM_CHARACTERISATION"))
    args = parser.parse_args()

    if args.intensities is not None:
        if any(number < 0 for number in args.intensities):
            parser.error("--intensities values must be zero or positive integers")
        if len(set(args.intensities)) != len(args.intensities):
            parser.error("--intensities must not contain duplicate values")

    args.output.mkdir(parents=True, exist_ok=True)
    bts, spectra, wavelengths = read_bts_xls(args.bts)
    sqm = read_sqm_files(args.data_dir, args.pattern)
    all_matched = pd.merge(sqm, bts, on="intensity", how="inner", validate="one_to_one")
    if all_matched.empty:
        raise ValueError("No matching intensity numbers between SQM filenames and BTS comments")

    available_matched = set(all_matched.intensity.astype(int))
    if args.intensities is None:
        selected_intensities = sorted(available_matched)
    else:
        selected_intensities = list(args.intensities)
        unavailable_requested = sorted(set(selected_intensities) - available_matched)
        if unavailable_requested:
            raise ValueError(
                "Requested intensities are not available in both BTS and SQM data: "
                f"{unavailable_requested}. Available matched intensities: "
                f"{sorted(available_matched)}"
            )

    table = all_matched.loc[
        all_matched.intensity.astype(int).isin(selected_intensities)
    ].copy()
    order_map = {number: position for position, number in enumerate(selected_intensities)}
    table["_selected_order"] = table.intensity.astype(int).map(order_map)
    table = table.sort_values("_selected_order").drop(columns="_selected_order")

    if len(table) < 2:
        raise ValueError(
            "At least two selected, matched intensity levels are required for the fixed-slope fit"
        )

    excluded_matched = sorted(available_matched - set(selected_intensities))
    missing_sqm = sorted(set(bts.intensity) - set(sqm.intensity))
    missing_bts = sorted(set(sqm.intensity) - set(bts.intensity))
    if missing_sqm:
        print("Warning: BTS levels without SQM CSV:", missing_sqm)
    if missing_bts:
        print("Warning: SQM levels without BTS comment:", missing_bts)

    integrated = []
    spectral_rows = []
    for _, row in table.iterrows():
        number = int(row.intensity)
        irradiance = spectra[number]
        radiance = irradiance / math.pi
        in_band = (wavelengths >= 360) & (wavelengths <= 830)
        value = trapz(radiance[in_band], wavelengths[in_band])
        integrated.append(value)
        for wl, e, l in zip(wavelengths, irradiance, radiance):
            spectral_rows.append({"intensity": number, "wavelength_nm": wl,
                                  "spectral_irradiance_W_m2_nm": e,
                                  "spectral_radiance_W_m2_sr_nm": l})

    table["integrated_radiance_360_830_W_m2_sr"] = integrated
    table["sqm_relative_signal"] = 10 ** (-0.4 * table.sqm_mpsas)
    table = table.sort_values("intensity").reset_index(drop=True)

    predictor = "integrated_radiance_360_830_W_m2_sr"
    radiance_model = fit_fixed_magnitude_model(table, predictor)
    lux_model = fit_fixed_magnitude_model(table, "illuminance_lx")
    table["sqm_mpsas_predicted_fixed_slope"] = (
        radiance_model["intercept_a"]
        - 2.5 * np.log10(table[predictor])
    )
    table["sqm_magnitude_residual"] = (
        table["sqm_mpsas"] - table["sqm_mpsas_predicted_fixed_slope"]
    )
    table["individual_intercept_a"] = (
        table["sqm_mpsas"] + 2.5 * np.log10(table[predictor])
    )
    table["individual_L0_W_m2_sr"] = 10 ** (
        table["individual_intercept_a"] / 2.5
    )

    make_figures(table, spectra, wavelengths, args.output, radiance_model, predictor)

    table.to_csv(args.output / "sqm_characterisation_summary.csv", index=False)
    pd.DataFrame(spectral_rows).to_csv(args.output / "bts_spectral_radiance_long.csv", index=False)
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Mean of final 3 valid SQM magnitudes; BTS irradiance divided by pi; fit m = a - 2.5 log10(Le); derive L0 = 10^(a/2.5)",
        "bts_file": str(args.bts.resolve()),
        "sqm_data_dir": str(args.data_dir.resolve()),
        "selected_intensities": selected_intensities,
        "used_matched_intensities": table.intensity.astype(int).tolist(),
        "excluded_matched_intensities": excluded_matched,
        "unmatched_bts_intensities": missing_sqm,
        "unmatched_sqm_intensities": missing_bts,
        "radiance_model": radiance_model,
        "illuminance_model": lux_model,
        "notes": [
            "The radiance model is the primary characterization.",
            "The lux model is secondary because illuminance is not the SQM measurand.",
            "Changing incandescent-lamp intensity changes spectral shape, so this characterisation is source-specific.",
            "With few intensity levels, fit statistics are descriptive and should not be treated as a full uncertainty evaluation.",
        ],
    }
    (args.output / "fit_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Selected intensity level(s): {selected_intensities}")
    print(f"Used {len(table)} matched intensity level(s): {table.intensity.astype(int).tolist()}")
    if excluded_matched:
        print(f"Excluded matched intensity level(s): {excluded_matched}")
    print(
        f"Fitted model: m = {radiance_model['intercept_a']:.8g} "
        "- 2.5 log10(Le)"
    )
    print(
        "Equivalent model: m = -2.5 log10(Le / L0), "
        f"L0 = {radiance_model['reference_radiance_L0']:.8e} "
        "W m^-2 sr^-1"
    )
    print(
        f"R2 = {radiance_model['r2_fixed_slope']:.6f}; "
        f"RMSE = {radiance_model['rmse_magnitude']:.6g} mag"
    )
    print("Results saved to", args.output.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
