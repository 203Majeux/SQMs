#!/usr/bin/env python3
"""Batch-characterise several SQMs from measurement subfolders.

Each positional folder must contain ``sqmX`` somewhere in its folder name,
where X is the SQM number. The number selects the column
``SQM_X_relative_spectral_response`` from the responsivity CSV.

Examples
--------
python characterise_multiple_SQMs_v2.py \
    DATA/with_housing/sqm1 DATA/with_housing/sqm2 DATA/with_housing/sqm3 \
    --bts-files REFERENCES/BTS/ref_sqm1.xls ref_sqm2.xls ref_sqm3.xls \
    --intensities "all, 1 2 3, 1 2 3 5"

The SQM folders, BTS files, and intensity groups are paired by position:
    sqm1 folder <-> first BTS file <-> first intensity group
    sqm2 folder <-> second BTS file <-> second intensity group
    sqm3 folder <-> third BTS file <-> third intensity group

    
Outputs
-------
RESULTS_MULTI_SQM_CHARACTERISATION/
    all_SQMs_responsivity_vs_CIE_photopic.png
    SQM_calibration_constants.csv
    SQM_calibration_constants.json
    sqm1/
        L0_magnitude_fit.png
        G_frequency_fit.png
        characterisation_by_intensity.csv
        weighted_spectra_long.csv
        characterisation_results.json
    sqm2/ ...
"""
from __future__ import annotations

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
SQM_FOLDER_RE = re.compile(r"sqm\s*[_-]?\s*(\d+)", re.IGNORECASE)


def trapz(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.trapezoid(y[valid], x[valid])) if valid.sum() >= 2 else np.nan


def intensity_number(value):
    match = INTENSITY_RE.search(str(value))
    return int(match.group(1)) if match else None


def sqm_number_from_folder(folder: Path):
    """Extract X from sqmX in the final folder name."""
    match = SQM_FOLDER_RE.search(folder.name)
    if not match:
        raise ValueError(
            f"Folder name must contain sqmX, for example sqm3: {folder}"
        )
    return int(match.group(1))


def circular_port_geometry(diameter_m, distance_m):
    """Exact projected-solid-angle geometry for a uniform circular port."""
    if diameter_m <= 0 or distance_m <= 0:
        raise ValueError("Aperture diameter and distance must be positive")
    radius = diameter_m / 2.0
    projected = math.pi * radius**2 / (distance_m**2 + radius**2)
    geometric = 2.0 * math.pi * (
        1.0 - distance_m / math.sqrt(distance_m**2 + radius**2)
    )
    half_angle = math.atan2(radius, distance_m)
    return {
        "diameter_m": diameter_m,
        "radius_m": radius,
        "distance_m": distance_m,
        "half_angle_deg": math.degrees(half_angle),
        "full_angle_deg": 2.0 * math.degrees(half_angle),
        "projected_solid_angle_sr": projected,
        "geometric_solid_angle_sr": geometric,
        "radiance_multiplier_sr_inverse": 1.0 / projected,
    }


def read_responsivity_table(path):
    frame = pd.read_csv(path)
    if "Wavelength_nm" not in frame:
        raise ValueError(f"Missing Wavelength_nm in {path}")
    return frame


def selected_responsivity(frame, sqm_number):
    column = f"SQM_{sqm_number}_relative_spectral_response"
    if column not in frame:
        available = [c for c in frame if c.endswith("_relative_spectral_response")]
        raise ValueError(
            f"Missing {column}. Available responsivity columns: {available}"
        )
    data = pd.DataFrame({
        "wavelength_nm": pd.to_numeric(frame["Wavelength_nm"], errors="coerce"),
        "response": pd.to_numeric(frame[column], errors="coerce"),
    }).dropna()
    data = (
        data.groupby("wavelength_nm", as_index=False)
        .response.mean()
        .sort_values("wavelength_nm")
    )
    if len(data) < 3 or data.response.max() <= 0:
        raise ValueError(f"Not enough valid values in {column}")
    response = np.clip(data.response.to_numpy(float), 0, None)
    response /= response.max()
    return data.wavelength_nm.to_numpy(float), response, column


def read_cie(path):
    frame = pd.read_csv(path, header=None, comment="#")
    if frame.shape[1] < 2:
        raise ValueError(f"CIE file requires two columns: {path}")
    wavelength = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(float)
    valid = np.isfinite(wavelength) & np.isfinite(values)
    order = np.argsort(wavelength[valid])
    return wavelength[valid][order], values[valid][order]


def read_bts_xls(path):
    try:
        raw = pd.read_excel(path, header=None, engine="calamine")
    except ImportError as exc:
        raise ImportError(
            "Legacy XLS support requires python-calamine. Install with: "
            "python -m pip install python-calamine"
        ) from exc

    labels = raw.iloc[:, 0].astype(str).str.strip().str.lower()

    def find_row(label, last=False):
        matches = raw.index[labels.eq(label.lower())].tolist()
        if not matches:
            raise ValueError(f"BTS row not found: {label!r}")
        return matches[-1] if last else matches[0]

    comment_row = find_row("comment")
    lux_row = find_row("integral sensor value")
    wavelength_header = find_row("wavelength /nm", last=True)
    cct_matches = raw.index[labels.eq("cct")].tolist()
    cct_row = cct_matches[0] if cct_matches else None

    columns = {}
    records = []
    for column in range(1, raw.shape[1]):
        intensity = intensity_number(raw.iat[comment_row, column])
        if intensity is None:
            continue
        if intensity in columns:
            raise ValueError(f"Duplicate BTS intensity: {intensity}")
        columns[intensity] = column
        records.append({
            "intensity": intensity,
            "bts_comment": str(raw.iat[comment_row, column]).strip(),
            "illuminance_lx": pd.to_numeric(
                raw.iat[lux_row, column], errors="coerce"
            ),
            "cct_K": (
                pd.to_numeric(raw.iat[cct_row, column], errors="coerce")
                if cct_row is not None else np.nan
            ),
        })

    block = raw.iloc[wavelength_header + 1:].copy()
    wavelength_series = pd.to_numeric(block.iloc[:, 0], errors="coerce")
    valid_wavelength = wavelength_series.between(100, 3000)
    wavelength = wavelength_series.loc[valid_wavelength].to_numpy(float)
    spectra = {}
    for intensity, column in columns.items():
        spectra[intensity] = pd.to_numeric(
            block.iloc[:, column], errors="coerce"
        ).loc[valid_wavelength].to_numpy(float)
    order = np.argsort(wavelength)
    return (
        pd.DataFrame(records),
        {key: values[order] for key, values in spectra.items()},
        wavelength[order],
    )


def read_measurement_folder(folder, file_pattern):
    files = sorted(folder.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No measurement CSV matches {folder / file_pattern}")
    records = []
    for path in files:
        intensity = intensity_number(path.stem)
        if intensity is None:
            continue
        frame = pd.read_csv(path)
        if "mpsas" not in frame:
            continue
        magnitudes = pd.to_numeric(frame["mpsas"], errors="coerce").dropna()
        if magnitudes.empty:
            continue
        magnitude_tail = magnitudes.tail(3)
        frequencies = (
            pd.to_numeric(frame["frequency_hz"], errors="coerce").dropna()
            if "frequency_hz" in frame else pd.Series(dtype=float)
        )
        frequency_tail = frequencies.tail(3)
        temperatures = (
            pd.to_numeric(frame["temperature_c"], errors="coerce").dropna()
            if "temperature_c" in frame else pd.Series(dtype=float)
        )
        records.append({
            "intensity": intensity,
            "measurement_file": str(path),
            "sqm_mpsas": float(magnitude_tail.mean()),
            "sqm_mpsas_std": (
                float(magnitude_tail.std(ddof=1))
                if len(magnitude_tail) > 1 else np.nan
            ),
            "sqm_frequency_Hz": (
                float(frequency_tail.mean()) if len(frequency_tail) else np.nan
            ),
            "sqm_frequency_std_Hz": (
                float(frequency_tail.std(ddof=1))
                if len(frequency_tail) > 1 else np.nan
            ),
            "sqm_temperature_C": (
                float(temperatures.tail(3).mean())
                if len(temperatures) else np.nan
            ),
            "n_magnitude_values": int(len(magnitudes)),
            "n_values_averaged": int(len(magnitude_tail)),
            "zero_magnitude_flag": bool((magnitudes == 0).any()),
        })
    if not records:
        raise ValueError(f"No usable SQM measurements in {folder}")
    result = pd.DataFrame(records)
    duplicates = result.intensity[result.intensity.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate intensity CSVs in {folder}: {duplicates}")
    return result


def fixed_magnitude_fit(quantity, magnitude):
    quantity = np.asarray(quantity, float)
    magnitude = np.asarray(magnitude, float)
    valid = np.isfinite(quantity) & (quantity > 0) & np.isfinite(magnitude)
    quantity = quantity[valid]
    magnitude = magnitude[valid]
    if len(quantity) < 2:
        raise ValueError("At least two levels are required for the L0 fit")
    intercept = float(np.mean(magnitude + 2.5 * np.log10(quantity)))
    zero_quantity = float(10 ** (intercept / 2.5))
    predicted = intercept - 2.5 * np.log10(quantity)
    residual = magnitude - predicted
    return {
        "n": int(len(quantity)),
        "intercept_a": intercept,
        "L0_W_m2_sr": zero_quantity,
        "rmse_mag": float(np.sqrt(np.mean(residual**2))),
        "residual_std_mag": (
            float(np.std(residual, ddof=1)) if len(quantity) > 1 else np.nan
        ),
    }


def frequency_fit(radiance, frequency, fixed_dark_frequency=None):
    """Fit f = K*L + fD with the physical constraints K > 0 and fD >= 0.

    When no independently measured dark frequency is supplied, an
    unconstrained straight-line fit is first calculated for diagnostics. If
    its intercept is negative, the constrained least-squares solution lies at
    the boundary fD = 0, so K is refitted through that boundary.

    When fixed_dark_frequency is supplied, that nonnegative value is used and
    only K is fitted.
    """
    radiance = np.asarray(radiance, float)
    frequency = np.asarray(frequency, float)
    valid = (
        np.isfinite(radiance)
        & (radiance >= 0)
        & np.isfinite(frequency)
        & (frequency >= 0)
    )
    radiance = radiance[valid]
    frequency = frequency[valid]
    if len(radiance) < 2:
        raise ValueError("At least two frequency values are required for G")
    if np.sum(radiance**2) <= 0:
        raise ValueError("Radiance values do not permit a frequency fit")

    # Always retain the unconstrained solution as a diagnostic.
    K_free, intercept_free = np.polyfit(radiance, frequency, 1)

    if fixed_dark_frequency is not None:
        if fixed_dark_frequency < 0:
            raise ValueError("A physical fixed dark frequency must be >= 0 Hz")
        fD = float(fixed_dark_frequency)
        K = float(
            np.sum(radiance * (frequency - fD))
            / np.sum(radiance**2)
        )
        method = "fixed independently supplied dark frequency"
        at_lower_bound = bool(np.isclose(fD, 0.0, atol=1e-12))
        dark_frequency_source = "user supplied"
    elif intercept_free >= 0 and K_free > 0:
        K = float(K_free)
        fD = float(intercept_free)
        method = "nonnegative free intercept"
        at_lower_bound = False
        dark_frequency_source = "constrained regression"
    else:
        # For linear least squares with fD >= 0, a negative unconstrained
        # intercept places the optimum on the boundary fD = 0.
        fD = 0.0
        K = float(
            np.sum(radiance * frequency)
            / np.sum(radiance**2)
        )
        method = "nonnegative constrained fit; fD at zero boundary"
        at_lower_bound = True
        dark_frequency_source = "constrained regression boundary"

    if K <= 0:
        raise ValueError(
            "The fitted frequency sensitivity K is not positive; calibration "
            "data are inconsistent with the physical model"
        )

    predicted = fD + K * radiance
    residual = frequency - predicted
    ss_res = float(np.sum(residual**2))
    ss_total = float(np.sum((frequency - frequency.mean())**2))
    r2 = 1.0 - ss_res / ss_total if ss_total > 0 else np.nan
    G = 1.0 / K

    # Diagnostic statistics for the unconstrained line.
    predicted_free = intercept_free + K_free * radiance
    residual_free = frequency - predicted_free
    ss_res_free = float(np.sum(residual_free**2))
    r2_free = 1.0 - ss_res_free / ss_total if ss_total > 0 else np.nan

    return {
        "n": int(len(radiance)),
        "method": method,
        "dark_frequency_source": dark_frequency_source,
        "K_Hz_per_W_m2_sr": float(K),
        "G_W_m2_sr_per_Hz": float(G),
        "fD_Hz": float(fD),
        "fD_at_lower_bound": at_lower_bound,
        "unconstrained_K_Hz_per_W_m2_sr": float(K_free),
        "unconstrained_intercept_Hz": float(intercept_free),
        "unconstrained_fit_R2": float(r2_free),
        "unconstrained_fit_RMSE_Hz": float(
            np.sqrt(np.mean(residual_free**2))
        ),
        "r2": float(r2),
        "rmse_Hz": float(np.sqrt(np.mean(residual**2))),
    }

def characterise_one_sqm(
    folder, sqm_number, response_wavelength, response, responsivity_column,
    bts_metadata, bts_spectra, bts_wavelength, geometry, intensities,
    fixed_dark_frequency, file_pattern, output_root,
):
    instrument = f"SQM{sqm_number}"
    output = output_root / f"sqm{sqm_number}"
    output.mkdir(parents=True, exist_ok=True)

    measurements = read_measurement_folder(folder, file_pattern)
    table = pd.merge(
        measurements, bts_metadata, on="intensity", how="inner",
        validate="one_to_one",
    )
    available = sorted(table.intensity.astype(int).tolist())
    selected = available if intensities is None else list(intensities)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(
            f"{instrument}: requested intensities unavailable in both datasets: "
            f"{missing}; available: {available}"
        )
    table = table[
        table.intensity.astype(int).isin(selected)
    ].sort_values("intensity").reset_index(drop=True)
    if len(table) < 2:
        raise ValueError(f"{instrument}: at least two matched levels are required")

    response_on_bts = np.interp(
        bts_wavelength, response_wavelength, response, left=0.0, right=0.0
    )
    omega_projected = geometry["projected_solid_angle_sr"]
    weighted_radiance = []
    spectral_rows = []

    for intensity in table.intensity.astype(int):
        spectral_irradiance = bts_spectra[intensity]
        spectral_radiance = spectral_irradiance / omega_projected
        sqm_radiance = trapz(
            spectral_radiance * response_on_bts, bts_wavelength
        )
        weighted_radiance.append(sqm_radiance)
        for wl, irradiance, radiance, weight in zip(
            bts_wavelength, spectral_irradiance,
            spectral_radiance, response_on_bts,
        ):
            spectral_rows.append({
                "instrument": instrument,
                "intensity": intensity,
                "wavelength_nm": wl,
                "spectral_irradiance_W_m2_nm": irradiance,
                "spectral_radiance_W_m2_sr_nm": radiance,
                "sqm_relative_responsivity": weight,
                "sqm_weighted_spectral_radiance": radiance * weight,
            })

    table["sqm_number"] = sqm_number
    table["projected_solid_angle_sr"] = omega_projected
    table["sqm_band_weighted_radiance_W_m2_sr"] = weighted_radiance

    l0_model = fixed_magnitude_fit(
        table.sqm_band_weighted_radiance_W_m2_sr, table.sqm_mpsas
    )
    table["predicted_mpsas_from_L0"] = (
        l0_model["intercept_a"]
        - 2.5 * np.log10(table.sqm_band_weighted_radiance_W_m2_sr)
    )
    table["magnitude_residual"] = (
        table.sqm_mpsas - table.predicted_mpsas_from_L0
    )
    table["individual_L0_W_m2_sr"] = (
        table.sqm_band_weighted_radiance_W_m2_sr
        * 10 ** (0.4 * table.sqm_mpsas)
    )

    g_model = frequency_fit(
        table.sqm_band_weighted_radiance_W_m2_sr,
        table.sqm_frequency_Hz,
        fixed_dark_frequency,
    )
    table["predicted_frequency_Hz"] = (
        g_model["fD_Hz"]
        + g_model["K_Hz_per_W_m2_sr"]
        * table.sqm_band_weighted_radiance_W_m2_sr
    )
    table["frequency_residual_Hz"] = (
        table.sqm_frequency_Hz - table.predicted_frequency_Hz
    )

    # L0 figure.
    radiance = table.sqm_band_weighted_radiance_W_m2_sr.to_numpy(float)
    magnitude = table.sqm_mpsas.to_numpy(float)
    radiance_line = np.geomspace(radiance.min(), radiance.max(), 400)
    magnitude_line = (
        l0_model["intercept_a"] - 2.5 * np.log10(radiance_line)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(radiance, magnitude, color="black", s=55, label="Measurements")
    ax.plot(
        radiance_line, magnitude_line,
        label=(
            rf"$m_{{\mathrm{{SQM}}}}=-2.5\log_{{10}}"
            rf"(L_{{\mathrm{{SQM}}}}/L_0)$"
            "\n"
            rf"$L_0={l0_model['L0_W_m2_sr']:.4e}\ "
            rf"\mathrm{{W\,m^{{-2}}\,sr^{{-1}}}}$"
        ),
    )
    for _, row in table.iterrows():
        ax.annotate(
            str(int(row.intensity)),
            (row.sqm_band_weighted_radiance_W_m2_sr, row.sqm_mpsas),
            xytext=(4, 4), textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set(
        xlabel=r"SQM-band-weighted radiance $L_{\mathrm{SQM}}$ (W m$^{-2}$ sr$^{-1}$)",
        ylabel=r"$m_{\mathrm{SQM}}$ (mag arcsec$^{-2}$)",
        title=f"{instrument} magnitude calibration",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "L0_magnitude_fit.png", dpi=220)
    plt.close(fig)

    # G figure with fit and residuals.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    frequency = table.sqm_frequency_Hz.to_numpy(float)
    line_radiance = np.linspace(radiance.min(), radiance.max(), 400)
    line_frequency = (
        g_model["fD_Hz"]
        + g_model["K_Hz_per_W_m2_sr"] * line_radiance
    )
    fd_label = (
        rf"$f_D={g_model['fD_Hz']:.4g}\ "
        rf"\mathrm{{Hz}}\ \mathrm{{(lower\ bound\ enforced)}}$"
        if g_model["fD_at_lower_bound"]
        else
        rf"$f_D={g_model['fD_Hz']:.4g}\ \mathrm{{Hz}}$"
    )
    axes[0].scatter(radiance, frequency, s=55, color="black")
    axes[0].plot(
        line_radiance,
        line_frequency,
        label=(
            rf"$L_{{\mathrm{{SQM}}}}=G(f-f_D)$"
            "\n"
            rf"$G={g_model['G_W_m2_sr_per_Hz']:.4e}\ "
            rf"\mathrm{{W\,m^{{-2}}\,sr^{{-1}}\,Hz^{{-1}}}}$"
            "\n"
            + fd_label
        ),
    )
    axes[0].set(
        xlabel=r"$L_{\mathrm{SQM}}$ (W m$^{-2}$ sr$^{-1}$)",
        ylabel="SQM frequency (Hz)",
        title=f"{instrument} frequency calibration",
    )
    axes[0].legend()
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].scatter(radiance, table.frequency_residual_Hz, s=55)
    axes[1].set(
        xlabel=r"$L_{\mathrm{SQM}}$ (W m$^{-2}$ sr$^{-1}$)",
        ylabel="Frequency residual (Hz)",
        title=f"Residuals, $R^2$={g_model['r2']:.6f}",
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "G_frequency_fit.png", dpi=220)
    plt.close(fig)

    table.to_csv(output / "characterisation_by_intensity.csv", index=False)
    pd.DataFrame(spectral_rows).to_csv(
        output / "weighted_spectra_long.csv", index=False
    )
    details = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instrument": instrument,
        "sqm_number": sqm_number,
        "measurement_folder": str(folder.resolve()),
        "responsivity_column": responsivity_column,
        "intensities": table.intensity.astype(int).tolist(),
        "geometry": geometry,
        "L0_model": l0_model,
        "frequency_model": g_model,
        "equations": {
            "spectral_radiance": (
                "L_lambda = E_lambda / Omega_projected"
            ),
            "projected_solid_angle": (
                "Omega_projected = pi*r^2/(d^2+r^2)"
            ),
            "weighted_radiance": (
                "L_SQM = integral L_lambda*T_SQM(lambda) dlambda"
            ),
            "magnitude": (
                "m_SQM = -2.5 log10(L_SQM/L0)"
            ),
            "frequency": "L_SQM = G*(f-fD)",
        },
    }
    (output / "characterisation_results.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    return {
        "sqm": f"sqm{sqm_number}",
        "sqm_number": sqm_number,
        "measurement_folder": str(folder.resolve()),
        "responsivity_column": responsivity_column,
        "n_intensities": int(len(table)),
        "intensities": ";".join(
            str(value) for value in table.intensity.astype(int)
        ),
        "L0_W_m2_sr": l0_model["L0_W_m2_sr"],
        "L0_fit_RMSE_mag": l0_model["rmse_mag"],
        "fD_Hz": g_model["fD_Hz"],
        "fD_at_lower_bound": g_model["fD_at_lower_bound"],
        "unconstrained_frequency_intercept_Hz": g_model[
            "unconstrained_intercept_Hz"
        ],
        "G_W_m2_sr_per_Hz": g_model["G_W_m2_sr_per_Hz"],
        "K_Hz_per_W_m2_sr": g_model["K_Hz_per_W_m2_sr"],
        "frequency_fit_R2": g_model["r2"],
        "frequency_fit_RMSE_Hz": g_model["rmse_Hz"],
        "projected_solid_angle_sr": omega_projected,
        "aperture_diameter_m": geometry["diameter_m"],
        "distance_m": geometry["distance_m"],
    }


def parse_intensity_groups(specification, expected_count):
    """Parse one comma-separated intensity group per SQM.

    Example for three SQMs:
        "all, 1 2 3, 1 2 3 5"

    ``all`` maps to None and means all matched intensities. Each other group
    is a whitespace-separated list of positive integer intensity numbers.
    """
    if specification is None:
        return [None] * expected_count

    groups = [group.strip() for group in specification.split(",")]
    if len(groups) != expected_count:
        raise ValueError(
            "The number of comma-separated --intensities groups must equal "
            f"the number of SQM folders ({expected_count}). Received "
            f"{len(groups)} group(s): {groups}"
        )

    parsed = []
    for position, group in enumerate(groups, start=1):
        if group.lower() == "all":
            parsed.append(None)
            continue
        if not group:
            raise ValueError(
                f"Intensity group {position} is empty. Use 'all' or provide "
                "space-separated integers."
            )
        try:
            values = [int(value) for value in group.split()]
        except ValueError as exc:
            raise ValueError(
                f"Invalid intensity group {position}: {group!r}. Use 'all' "
                "or space-separated integers."
            ) from exc
        if any(value < 0 for value in values):
            raise ValueError(
                f"Intensity group {position} contains a negative value: {values}"
            )
        if len(values) != len(set(values)):
            raise ValueError(
                f"Intensity group {position} contains duplicates: {values}"
            )
        parsed.append(values)
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the measured responsivities of selected SQMs together with "
            "the CIE photopic and scotopic response curves. No BTS reference "
            "file is required."
        )
    )
    parser.add_argument(
        "--sqm1", type=Path, default=None,
        help="SQM 1 characterisation CSV (used to select and label SQM 1)",
    )
    parser.add_argument(
        "--sqm2", type=Path, default=None,
        help="SQM 2 characterisation CSV (used to select and label SQM 2)",
    )
    parser.add_argument(
        "--sqm3", type=Path, default=None,
        help="SQM 3 characterisation CSV (used to select and label SQM 3)",
    )
    parser.add_argument(
        "--responsivity", type=Path,
        default=Path("sqmTotal_results/sqm_characterization.csv"),
        help=(
            "CSV containing Wavelength_nm and the SQM_X_relative_spectral_response "
            "columns"
        ),
    )
    parser.add_argument(
        "--cie-photopic", type=Path,
        default=Path("PhotometryCIE/CIE_sle_photopic.csv"),
    )
    parser.add_argument(
        "--cie-scotopic", type=Path,
        default=Path("PhotometryCIE/CIE_sle_scotopic.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("RESULTS_MULTI_SQM_CHARACTERISATION"),
    )
    args = parser.parse_args()

    selected_files = {
        number: path
        for number, path in ((1, args.sqm1), (2, args.sqm2), (3, args.sqm3))
        if path is not None
    }
    if not selected_files:
        parser.error("Provide at least one input, for example --sqm1 PATH")

    for number, path in selected_files.items():
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"SQM {number} CSV not found: {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    responsivity_table = read_responsivity_table(args.responsivity)
    cie_wavelength, cie_v = read_cie(args.cie_photopic)
    scotopic_wavelength, scotopic_v = read_cie(args.cie_scotopic)

    selected_curves = {}
    for number in sorted(selected_files):
        selected_curves[number] = selected_responsivity(
            responsivity_table, number
        )

    minimum_wavelength = min(
        [cie_wavelength.min(), scotopic_wavelength.min()]
        + [curve[0].min() for curve in selected_curves.values()]
    )
    maximum_wavelength = max(
        [cie_wavelength.max(), scotopic_wavelength.max()]
        + [curve[0].max() for curve in selected_curves.values()]
    )
    common_wavelength = np.arange(
        math.floor(minimum_wavelength),
        math.ceil(maximum_wavelength) + 1,
        1.0,
    )

    photopic_interpolated = np.interp(
        common_wavelength, cie_wavelength, cie_v,
        left=0.0, right=0.0,
    )
    scotopic_interpolated = np.interp(
        common_wavelength, scotopic_wavelength, scotopic_v,
        left=0.0, right=0.0,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        common_wavelength, photopic_interpolated,
        color="black", linewidth=2.5,
        label=r"CIE photopic $V(\lambda)$",
    )
    ax.plot(
        common_wavelength, scotopic_interpolated,
        color="tab:blue", linewidth=2.5, linestyle="--",
        label=r"CIE scotopic $V'(\lambda)$",
    )

    combined_response = {
        "wavelength_nm": common_wavelength,
        "cie_photopic_V_lambda": photopic_interpolated,
        "cie_scotopic_V_lambda": scotopic_interpolated,
    }

    for number in sorted(selected_curves):
        wavelength, response, _ = selected_curves[number]
        interpolated = np.interp(
            common_wavelength, wavelength, response,
            left=0.0, right=0.0,
        )
        line, = ax.plot(
            common_wavelength, interpolated,
            linewidth=2.0,
            label=f"SQM {number}",
        )
        # Show the original measured responsivity samples on the same line.
        ax.scatter(
            wavelength, response,
            color=line.get_color(),
            s=34, marker="o", edgecolors="white", linewidths=0.5,
            zorder=3,
        )
        combined_response[
            f"SQM_{number}_relative_responsivity"
        ] = interpolated

    ax.set_xlabel("Wavelength (nm)", fontsize=21)
    ax.set_ylabel("Normalized response", fontsize=21)
    # ax.set_title(
    #     "CIE photopic and scotopic responses with measured SQM responsivities",
    #     fontsize=18,
    # )
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)
    fig.tight_layout()

    figure_path = args.output / "all_SQMs_responsivity_vs_CIE_photopic.png"
    csv_path = args.output / "all_SQMs_responsivity_vs_CIE_photopic.csv"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    pd.DataFrame(combined_response).to_csv(csv_path, index=False)

    print("Saved figure:", figure_path.resolve())
    print("Saved combined data:", csv_path.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
