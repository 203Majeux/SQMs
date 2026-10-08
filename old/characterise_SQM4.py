#!/usr/bin/env python3
"""Characterise an SQM using its measured relative spectral responsivity.

Default layout
--------------
sqmTotal_results/sqm_characterization.csv
DATA/with_housing/sqm_measurements_SQM3_intensityX.csv
REFERENCES/BTS/ref_bts_with_minisphere_and_housing.xls
PhotometryCIE/CIE_sle_photopic.csv

Examples
--------
python characterise_SQM_with_responsivity.py sqm3
python characterise_SQM_with_responsivity.py sqm3 --intensities 1 2 3 4 5 6 7
python characterise_SQM_with_responsivity.py sqm1 --output RESULTS_SQM1_BAND

The selected SQM argument is case-insensitive. ``sqm3`` selects the column
``SQM_3_relative_spectral_response`` from the monochromator result file.
"""
from __future__ import annotations

import argparse
import csv
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
C_LIGHT = 299_792_458.0
AB_E0_NU_W_M2_HZ = 3631.0e-26
ARCSEC2_SR = (math.pi / (180.0 * 3600.0)) ** 2


def trapz(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.trapezoid(y[ok], x[ok])) if ok.sum() >= 2 else np.nan


def intensity_number(value):
    match = INTENSITY_RE.search(str(value))
    return int(match.group(1)) if match else None


def normalize_instrument(value: str) -> tuple[str, int]:
    clean = re.sub(r"[^a-zA-Z0-9]", "", value).upper()
    match = re.fullmatch(r"SQM([123])", clean)
    if not match:
        raise ValueError("Instrument must be SQM1, SQM2, or SQM3")
    number = int(match.group(1))
    return f"SQM{number}", number


def read_responsivity(path: Path, sqm_number: int):
    frame = pd.read_csv(path)
    wl_col = "Wavelength_nm"
    response_col = f"SQM_{sqm_number}_relative_spectral_response"
    if wl_col not in frame or response_col not in frame:
        raise ValueError(f"Required columns not found: {wl_col}, {response_col}")
    reduced = pd.DataFrame({
        "wavelength_nm": pd.to_numeric(frame[wl_col], errors="coerce"),
        "response": pd.to_numeric(frame[response_col], errors="coerce"),
    }).dropna()
    # Duplicate wavelengths exist in some IR measurement series; average them.
    reduced = reduced.groupby("wavelength_nm", as_index=False).response.mean().sort_values("wavelength_nm")
    reduced = reduced[(reduced.wavelength_nm >= 100) & (reduced.wavelength_nm <= 3000)]
    if len(reduced) < 3:
        raise ValueError(f"Too few valid responsivity values in {path}")
    response = reduced.response.to_numpy(float)
    if np.nanmax(response) <= 0:
        raise ValueError("Spectral response has no positive values")
    response = np.clip(response / np.nanmax(response), 0, None)
    return reduced.wavelength_nm.to_numpy(float), response


def read_cie(path: Path):
    frame = pd.read_csv(path, header=None, comment="#")
    if frame.shape[1] < 2:
        raise ValueError(f"CIE file requires two columns: {path}")
    wl = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(float)
    ok = np.isfinite(wl) & np.isfinite(values)
    order = np.argsort(wl[ok])
    return wl[ok][order], values[ok][order]


def read_bts_xls(path: Path):
    try:
        raw = pd.read_excel(path, header=None, engine="calamine")
    except ImportError as exc:
        raise ImportError("Install XLS support with: python -m pip install python-calamine") from exc
    labels = raw.iloc[:, 0].astype(str).str.strip().str.lower()

    def row(label, last=False):
        hits = raw.index[labels.eq(label.lower())].tolist()
        if not hits:
            raise ValueError(f"BTS row not found: {label!r}")
        return hits[-1] if last else hits[0]

    comment_row = row("comment")
    lux_row = row("integral sensor value")
    wave_header = row("wavelength /nm", last=True)
    cct_hits = raw.index[labels.eq("cct")].tolist()
    cct_row = cct_hits[0] if cct_hits else None

    columns, records = {}, []
    for col in range(1, raw.shape[1]):
        intensity = intensity_number(raw.iat[comment_row, col])
        if intensity is None:
            continue
        if intensity in columns:
            raise ValueError(f"Duplicate BTS intensity comment: intensity {intensity}")
        columns[intensity] = col
        records.append({
            "intensity": intensity,
            "bts_comment": str(raw.iat[comment_row, col]).strip(),
            "illuminance_lx": pd.to_numeric(raw.iat[lux_row, col], errors="coerce"),
            "cct_K": pd.to_numeric(raw.iat[cct_row, col], errors="coerce") if cct_row is not None else np.nan,
        })
    if not records:
        raise ValueError("No BTS comments matching intensityX were found")

    block = raw.iloc[wave_header + 1:].copy()
    wl_series = pd.to_numeric(block.iloc[:, 0], errors="coerce")
    valid_wl = wl_series.between(100, 3000)
    wl = wl_series.loc[valid_wl].to_numpy(float)
    spectra = {}
    for intensity, col in columns.items():
        spectra[intensity] = pd.to_numeric(block.iloc[:, col], errors="coerce").loc[valid_wl].to_numpy(float)
    order = np.argsort(wl)
    return pd.DataFrame(records), {k: v[order] for k, v in spectra.items()}, wl[order]


def read_sqm_files(data_dir: Path, instrument: str, pattern: str):
    files = sorted(data_dir.glob(pattern.format(instrument=instrument)))
    if not files:
        raise FileNotFoundError(f"No files match {data_dir / pattern.format(instrument=instrument)}")
    records = []
    for path in files:
        intensity = intensity_number(path.stem)
        if intensity is None:
            continue
        frame = pd.read_csv(path)
        if "mpsas" not in frame:
            continue
        mag = pd.to_numeric(frame.mpsas, errors="coerce").dropna()
        if mag.empty:
            continue
        mtail = mag.tail(3)
        frequency = pd.to_numeric(frame.get("frequency_hz"), errors="coerce").dropna() if "frequency_hz" in frame else pd.Series(dtype=float)
        ftail = frequency.tail(3)
        temperature = pd.to_numeric(frame.get("temperature_c"), errors="coerce").dropna() if "temperature_c" in frame else pd.Series(dtype=float)
        records.append({
            "intensity": intensity,
            "sqm_file": str(path),
            "sqm_n_total": int(len(mag)),
            "sqm_n_averaged": int(len(mtail)),
            "sqm_mpsas": float(mtail.mean()),
            "sqm_mpsas_std": float(mtail.std(ddof=1)) if len(mtail) > 1 else np.nan,
            "sqm_frequency_Hz": float(ftail.mean()) if len(ftail) else np.nan,
            "sqm_frequency_std_Hz": float(ftail.std(ddof=1)) if len(ftail) > 1 else np.nan,
            "sqm_temperature_C": float(temperature.tail(3).mean()) if len(temperature) else np.nan,
            "zero_magnitude_flag": bool((mag == 0).any()),
        })
    if not records:
        raise ValueError("No usable SQM measurement files")
    return pd.DataFrame(records)


def fixed_magnitude_fit(quantity, magnitude):
    q, m = np.asarray(quantity, float), np.asarray(magnitude, float)
    ok = np.isfinite(q) & (q > 0) & np.isfinite(m)
    q, m = q[ok], m[ok]
    if len(q) < 2:
        raise ValueError("At least two levels are required")
    a = float(np.mean(m + 2.5 * np.log10(q)))
    q0 = float(10 ** (a / 2.5))
    prediction = a - 2.5 * np.log10(q)
    residual = m - prediction
    return {"n": len(q), "intercept_a": a, "zero_quantity_q0": q0,
            "rmse_mag": float(np.sqrt(np.mean(residual**2))),
            "residual_std_mag": float(np.std(residual, ddof=1)) if len(q) > 1 else np.nan}


def free_fit(quantity, response):
    x, y = np.asarray(quantity, float), np.asarray(response, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return {"n": len(x), "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope*x
    ss_res, ss_tot = np.sum((y-pred)**2), np.sum((y-y.mean())**2)
    return {"n": len(x), "slope": float(slope), "intercept": float(intercept),
            "r2": float(1-ss_res/ss_tot) if ss_tot > 0 else np.nan}


def circular_port_geometry(diameter_m, distance_m):
    """Geometry for an on-axis, uniform circular Lambertian port.

    For an ideal cosine-response irradiance detector:
        E_lambda = L_lambda * Omega_projected
        Omega_projected = integral cos(theta) dOmega
                        = pi*r^2/(d^2+r^2)
    """
    if diameter_m <= 0 or distance_m <= 0:
        raise ValueError("Diameter and distance must be greater than zero")
    r = diameter_m / 2.0
    omega_projected = math.pi * r**2 / (distance_m**2 + r**2)
    omega_geometric = 2.0 * math.pi * (
        1.0 - distance_m / math.sqrt(distance_m**2 + r**2)
    )
    half_angle = math.atan2(r, distance_m)
    return {
        "diameter_m": diameter_m,
        "radius_m": r,
        "distance_m": distance_m,
        "half_angle_deg": math.degrees(half_angle),
        "full_angle_deg": 2.0 * math.degrees(half_angle),
        "projected_solid_angle_sr": omega_projected,
        "geometric_solid_angle_sr": omega_geometric,
        "radiance_multiplier_sr_inverse": 1.0 / omega_projected,
        "factor_relative_to_E_over_pi": math.pi / omega_projected,
    }


def main():
    parser = argparse.ArgumentParser(description="SQM-band characterization using measured spectral responsivity")
    parser.add_argument("instrument", help="SQM1, SQM2, or SQM3; sqm3 selects SQM_3_relative_spectral_response")
    parser.add_argument("--responsivity", type=Path, default=Path("sqmTotal_results/sqm_characterization.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/with_housing"))
    parser.add_argument("--bts", type=Path, default=Path("REFERENCES/BTS/ref_bts_with_minisphere_and_housing.xls"))
    parser.add_argument("--cie-photopic", type=Path, default=Path("PhotometryCIE/CIE_sle_photopic.csv"))
    parser.add_argument("--pattern", default="sqm_measurements_{instrument}_intensity*.csv")
    parser.add_argument("--intensities", nargs="+", type=int, default=None)
    parser.add_argument("--dark-frequency", type=float, default=None,
                        help="Known dark frequency in Hz. If omitted, it is fitted as the frequency intercept.")
    parser.add_argument("--aperture-diameter-m", type=float, default=0.05,
                        help="Sphere-port diameter in metres (default: 0.05)")
    parser.add_argument("--distance-m", type=float, default=0.12,
                        help="Port-to-reference-detector distance in metres (default: 0.12)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    instrument, number = normalize_instrument(args.instrument)
    geometry = circular_port_geometry(args.aperture_diameter_m, args.distance_m)
    omega_projected = geometry["projected_solid_angle_sr"]
    output = args.output or Path(f"RESULTS_{instrument}_SPECTRAL_CHARACTERISATION")
    output.mkdir(parents=True, exist_ok=True)

    response_wl, response = read_responsivity(args.responsivity, number)
    cie_wl, cie_v = read_cie(args.cie_photopic)
    bts, bts_spectra, bts_wl = read_bts_xls(args.bts)
    sqm = read_sqm_files(args.data_dir, instrument, args.pattern)
    table = pd.merge(sqm, bts, on="intensity", how="inner", validate="one_to_one")
    available = sorted(table.intensity.astype(int).tolist())
    selected = available if args.intensities is None else args.intensities
    missing = sorted(set(selected)-set(available))
    if missing:
        raise ValueError(f"Requested intensities unavailable in both datasets: {missing}; available: {available}")
    table = table[table.intensity.astype(int).isin(selected)].sort_values("intensity").reset_index(drop=True)
    if len(table) < 2:
        raise ValueError("Select at least two matched intensity levels")

    # Interpolate both weighting functions onto the BTS spectral grid.
    T = np.interp(bts_wl, response_wl, response, left=0.0, right=0.0)
    V = np.interp(bts_wl, cie_wl, cie_v, left=0.0, right=0.0)
    weighted_radiance, luminance, radiance_total, mismatch_ratio = [], [], [], []
    spectral_long = []
    for intensity in table.intensity.astype(int):
        E_lambda = bts_spectra[intensity]
        # Exact irradiance-to-radiance conversion for the stated geometry.
        L_lambda = E_lambda / omega_projected
        L_sqm = trapz(L_lambda*T, bts_wl)
        Lv = 683.0*trapz(L_lambda*V, bts_wl)
        Le = trapz(L_lambda, bts_wl)
        weighted_radiance.append(L_sqm)
        luminance.append(Lv)
        radiance_total.append(Le)
        mismatch_ratio.append(L_sqm/(Lv/683.0) if Lv > 0 else np.nan)
        for wl, e, l, t, v in zip(bts_wl, E_lambda, L_lambda, T, V):
            spectral_long.append({"intensity": intensity, "wavelength_nm": wl,
                                  "spectral_irradiance_W_m2_nm": e,
                                  "spectral_radiance_W_m2_sr_nm": l,
                                  "sqm_relative_response": t, "cie_V_lambda": v,
                                  "sqm_weighted_spectral_radiance": l*t,
                                  "photopic_weighted_spectral_radiance": l*v})

    table["aperture_diameter_m"] = geometry["diameter_m"]
    table["distance_m"] = geometry["distance_m"]
    table["projected_solid_angle_sr"] = omega_projected
    table["geometric_solid_angle_sr"] = geometry["geometric_solid_angle_sr"]
    table["sqm_band_weighted_radiance_W_m2_sr"] = weighted_radiance
    table["photopic_luminance_cd_m2"] = luminance
    table["unweighted_radiance_W_m2_sr"] = radiance_total
    table["sqm_to_photopic_radiometric_weight_ratio"] = mismatch_ratio

    mag_fit = fixed_magnitude_fit(table.sqm_band_weighted_radiance_W_m2_sr, table.sqm_mpsas)
    photopic_fit = fixed_magnitude_fit(table.photopic_luminance_cd_m2, table.sqm_mpsas)
    table["sqm_band_predicted_mpsas"] = (mag_fit["intercept_a"]
        - 2.5*np.log10(table.sqm_band_weighted_radiance_W_m2_sr))
    table["sqm_band_magnitude_residual"] = table.sqm_mpsas-table.sqm_band_predicted_mpsas
    table["individual_SQM_band_L0_W_m2_sr"] = (table.sqm_band_weighted_radiance_W_m2_sr
        * 10**(0.4*table.sqm_mpsas))

    # Frequency-to-band-radiance calibration. If fD is known, constrain through fD.
    freq_ok = np.isfinite(table.sqm_frequency_Hz) & np.isfinite(table.sqm_band_weighted_radiance_W_m2_sr)
    if freq_ok.sum() < 2:
        frequency_model = {"available": False}
    else:
        L = table.loc[freq_ok, "sqm_band_weighted_radiance_W_m2_sr"].to_numpy(float)
        f = table.loc[freq_ok, "sqm_frequency_Hz"].to_numpy(float)
        if args.dark_frequency is None:
            fit = free_fit(L, f)
            K, fD = fit["slope"], fit["intercept"]
            r2 = fit["r2"]
            method = "fitted f = K*L_SQM + fD"
        else:
            fD = float(args.dark_frequency)
            K = float(np.sum(L*(f-fD))/np.sum(L**2))
            pred = fD+K*L
            ss_res, ss_tot = np.sum((f-pred)**2), np.sum((f-f.mean())**2)
            r2 = float(1-ss_res/ss_tot) if ss_tot > 0 else np.nan
            method = "fixed dark frequency; fitted slope"
        G = 1.0/K if K > 0 else np.nan
        frequency_model = {"available": True, "method": method, "K_Hz_per_W_m2_sr": K,
                           "G_W_m2_sr_per_Hz": G, "dark_frequency_Hz": fD, "r2": r2}
        table["frequency_predicted_Hz"] = fD+K*table.sqm_band_weighted_radiance_W_m2_sr
        table["frequency_residual_Hz"] = table.sqm_frequency_Hz-table.frequency_predicted_Hz

    # AB reference radiance for the selected, normalized SQM band.
    wl_m = response_wl*1e-9
    L0_lambda_per_m = AB_E0_NU_W_M2_HZ*C_LIGHT/(wl_m**2*ARCSEC2_SR)
    Lr_AB = trapz(response*L0_lambda_per_m, wl_m)
    if frequency_model.get("available") and np.isfinite(frequency_model["G_W_m2_sr_per_Hz"]):
        G = frequency_model["G_W_m2_sr_per_Hz"]
        ZP_AB = 2.5*math.log10(Lr_AB/G)
        frequency_model["AB_reference_radiance_W_m2_sr"] = Lr_AB
        frequency_model["AB_frequency_zero_point_mag_arcsec2"] = ZP_AB
        table["sqm_AB_magnitude_from_frequency"] = (ZP_AB
            - 2.5*np.log10(np.maximum(table.sqm_frequency_Hz-frequency_model["dark_frequency_Hz"], np.nan)))
        table["AB_minus_factory_magnitude"] = table.sqm_AB_magnitude_from_frequency-table.sqm_mpsas
    else:
        ZP_AB = np.nan

    # Figure 1: selected SQM response versus CIE photopic curve.
    common = np.arange(math.floor(min(response_wl.min(), cie_wl.min())),
                       math.ceil(max(response_wl.max(), cie_wl.max()))+1, 1.0)
    response_common = np.interp(common, response_wl, response, left=0, right=0)
    cie_common = np.interp(common, cie_wl, cie_v, left=0, right=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(common, response_common, label=f"{instrument} measured relative responsivity")
    ax.plot(common, cie_common, label=r"CIE photopic $V(\lambda)$")
    ax.fill_between(common, response_common, cie_common, alpha=0.15, label="Spectral mismatch")
    ax.set(xlabel="Wavelength (nm)", ylabel="Normalized response",
           title=f"{instrument} spectral responsivity versus CIE photopic response")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(output/"01_SQM_responsivity_vs_CIE_photopic.png", dpi=220); plt.close(fig)

    # Figure 2: SQM-band magnitude response.
    q = table.sqm_band_weighted_radiance_W_m2_sr.to_numpy(float)
    m = table.sqm_mpsas.to_numpy(float)
    qline = np.geomspace(q.min(), q.max(), 300)
    mline = mag_fit["intercept_a"]-2.5*np.log10(qline)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(q, m, color="black", s=55, label="Measurements")
    ax.plot(qline, mline, label=(rf"$m_{{\mathrm{{SQM}}}}=-2.5\log_{{10}}(L_{{\mathrm{{SQM}}}}/L_0)$"
        "\n"+rf"$L_0={mag_fit['zero_quantity_q0']:.4e}\ \mathrm{{W\,m^{{-2}}\,sr^{{-1}}}}$"))
    for _, row in table.iterrows(): ax.annotate(str(int(row.intensity)), (row.sqm_band_weighted_radiance_W_m2_sr,row.sqm_mpsas), xytext=(4,4),textcoords="offset points")
    ax.set_xscale("log"); ax.invert_yaxis(); ax.grid(True,which="both",alpha=.3)
    ax.set(xlabel=r"SQM-band weighted radiance $L_{\mathrm{SQM}}$ (W m$^{-2}$ sr$^{-1}$)",
           ylabel=r"$m_{\mathrm{SQM}}$ (mag arcsec$^{-2}$)", title=f"{instrument} band-weighted magnitude response")
    ax.legend(); fig.tight_layout(); fig.savefig(output/"02_SQM_band_magnitude_response.png",dpi=220); plt.close(fig)

    # Figure 3: frequency calibration and residuals.
    if frequency_model.get("available"):
        fig, axes = plt.subplots(1,2,figsize=(13,5))
        axes[0].scatter(q, table.sqm_frequency_Hz, s=55)
        qf=np.linspace(q.min(),q.max(),300)
        axes[0].plot(qf,frequency_model["dark_frequency_Hz"]+frequency_model["K_Hz_per_W_m2_sr"]*qf,
                     label=rf"$L_{{SQM}}=G(f-f_D)$; $G={frequency_model['G_W_m2_sr_per_Hz']:.4e}$")
        axes[0].set(xlabel=r"$L_{\mathrm{SQM}}$ (W m$^{-2}$ sr$^{-1}$)",ylabel="SQM frequency (Hz)",title="Absolute frequency calibration")
        axes[0].legend()
        axes[1].axhline(0,color="black",lw=1);axes[1].scatter(q,table.frequency_residual_Hz,s=55)
        axes[1].set(xlabel=r"$L_{\mathrm{SQM}}$ (W m$^{-2}$ sr$^{-1}$)",ylabel="Frequency residual (Hz)",title=f"Frequency fit residuals, R2={frequency_model['r2']:.6f}")
        for ax in axes: ax.grid(True,alpha=.3)
        fig.tight_layout();fig.savefig(output/"03_frequency_radiance_calibration.png",dpi=220);plt.close(fig)

    # Figure 4: source-dependent mismatch and photopic zero point.
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    axes[0].plot(table.intensity,table.sqm_to_photopic_radiometric_weight_ratio,"o-")
    axes[0].set(xlabel="Intensity",ylabel=r"$L_{SQM}/(L_v/683)$",title="SQM-to-photopic spectral weighting ratio")
    axes[1].plot(table.intensity,table.individual_SQM_band_L0_W_m2_sr,"o-")
    axes[1].axhline(mag_fit["zero_quantity_q0"],ls="--",label="Fitted SQM-band zero radiance")
    axes[1].set(xlabel="Intensity",ylabel=r"Individual $L_0$ (W m$^{-2}$ sr$^{-1}$)",title="SQM-band zero-point stability")
    axes[1].legend()
    for ax in axes: ax.grid(True,alpha=.3)
    fig.tight_layout();fig.savefig(output/"04_spectral_mismatch_and_zero_point.png",dpi=220);plt.close(fig)

    table.to_csv(output/"characterisation_by_intensity.csv",index=False)
    pd.DataFrame(spectral_long).to_csv(output/"weighted_spectra_long.csv",index=False)
    pd.DataFrame({"wavelength_nm":common,"sqm_relative_responsivity":response_common,
                  "cie_photopic_V_lambda":cie_common}).to_csv(output/"SQM_vs_CIE_response.csv",index=False)

    results = {"created_utc":datetime.now(timezone.utc).isoformat(),"instrument":instrument,
               "selected_responsivity_column":f"SQM_{number}_relative_spectral_response",
               "selected_intensities":table.intensity.astype(int).tolist(),
               "inputs":{"responsivity":str(args.responsivity.resolve()),"data_dir":str(args.data_dir.resolve()),
                         "bts":str(args.bts.resolve()),"cie_photopic":str(args.cie_photopic.resolve())},
               "sqm_band_magnitude_model":mag_fit,"photopic_luminance_model":photopic_fit,
               "frequency_calibration":frequency_model,"AB_reference_radiance_W_m2_sr":Lr_AB,
               "measurement_geometry": geometry,
               "irradiance_to_radiance_equation": "L_lambda = E_lambda / [pi*r^2/(d^2+r^2)]",
               "normalization_note":"T_SQM(lambda) is normalized to 1 at its maximum, matching the paper convention.",
               "equations":{"sqm_band_radiance":"L_SQM = integral L_lambda*T_SQM(lambda) dlambda",
                            "frequency":"L_SQM = G*(f-fD)",
                            "magnitude":"m_SQM = -2.5 log10(L_SQM/L0)",
                            "AB":"m_AB = ZP_AB - 2.5 log10(f-fD)"}}
    (output/"characterisation_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")

    print(f"Instrument: {instrument}")
    print(f"Geometry: D={geometry['diameter_m']:.6g} m, d={geometry['distance_m']:.6g} m, full angle={geometry['full_angle_deg']:.6g} deg")
    print(f"Projected solid angle = {omega_projected:.10e} sr")
    print(f"Geometric solid angle = {geometry['geometric_solid_angle_sr']:.10e} sr (reported only)")
    print(f"Radiance conversion: L_lambda = E_lambda / {omega_projected:.10e}")
    print(f"Correction versus E_lambda/pi = x{geometry['factor_relative_to_E_over_pi']:.8g}")
    print(f"Responsivity column: SQM_{number}_relative_spectral_response")
    print(f"Used intensities: {table.intensity.astype(int).tolist()}")
    print(f"SQM-band L0 = {mag_fit['zero_quantity_q0']:.8e} W m^-2 sr^-1")
    print(f"SQM-band magnitude RMSE = {mag_fit['rmse_mag']:.6g} mag")
    print(f"Photopic Lv0 = {photopic_fit['zero_quantity_q0']:.8e} cd m^-2")
    print(f"AB reference radiance = {Lr_AB:.8e} W m^-2 sr^-1")
    if frequency_model.get("available"):
        print(f"G = {frequency_model['G_W_m2_sr_per_Hz']:.8e} W m^-2 sr^-1 Hz^-1")
        print(f"fD = {frequency_model['dark_frequency_Hz']:.6g} Hz")
        print(f"Frequency fit R2 = {frequency_model['r2']:.8f}")
        print(f"AB frequency zero point = {ZP_AB:.6f} mag arcsec^-2")
    print("Saved results to",output.resolve())
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}",file=sys.stderr)
        raise
