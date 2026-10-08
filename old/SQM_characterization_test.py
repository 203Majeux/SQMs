#!/usr/bin/env python3
"""Independent diagnostic checks for an SQM/BTS characterisation.

This companion script uses the readers and conventions in characterise_SQM2.py,
but writes a separate CHECKS_SQM_CHARACTERISATION folder. It does not modify the
original characterisation results.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("sqm_characterisation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def free_fit(luminance, magnitude):
    x, y = np.log10(np.asarray(luminance, float)), np.asarray(magnitude, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    x, y = x[ok], y[ok]
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res, ss_tot = np.sum((y-pred)**2), np.sum((y-y.mean())**2)
    return {"n": len(x), "slope": float(slope), "intercept": float(intercept),
            "r2": float(1-ss_res/ss_tot) if ss_tot > 0 else np.nan}


def main():
    ap = argparse.ArgumentParser(description="Check SQM zero-point characterisation assumptions")
    ap.add_argument("--characterisation-script", type=Path, default=Path("characterise_SQM2.py"))
    ap.add_argument("--bts", type=Path, default=Path("REFERENCES/BTS/ref_bts_with_minisphere.xls"))
    ap.add_argument("--cie-photopic", type=Path, default=Path("PhotometryCIE/CIE_sle_photopic.csv"))
    ap.add_argument("--data-dir", type=Path, default=Path("DATA"))
    ap.add_argument("--pattern", default="sqm_measurements_*_intensity*.csv")
    ap.add_argument("--intensities", nargs="+", type=int, default=None)
    ap.add_argument("--literature-L0", type=float, default=108000.0)
    ap.add_argument("--tolerance-percent", type=float, default=5.0)
    ap.add_argument("--output", type=Path, default=Path("CHECKS_SQM_CHARACTERISATION"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mod = load_module(args.characterisation_script)
    cie_wl, cie_v = mod.read_cie_photopic(args.cie_photopic)
    bts, spectra, wl = mod.read_bts_xls(args.bts)
    sqm = mod.read_sqm_files(args.data_dir, args.pattern)
    matched = pd.merge(sqm, bts, on="intensity", how="inner", validate="one_to_one")
    if args.intensities is not None:
        missing = sorted(set(args.intensities)-set(matched.intensity.astype(int)))
        if missing:
            raise ValueError(f"Requested intensities missing from matched data: {missing}")
        matched = matched[matched.intensity.astype(int).isin(args.intensities)].copy()
    matched = matched.sort_values("intensity").reset_index(drop=True)
    if len(matched) < 2:
        raise ValueError("At least two matched intensity levels are required")

    # Re-open SQM files to independently collect final-three frequency and saturation flags.
    frequencies, zero_flags = [], []
    for filename in matched.sqm_file:
        f = pd.read_csv(filename)
        frequencies.append(float(pd.to_numeric(f.get("frequency_hz"), errors="coerce").tail(3).mean())
                           if "frequency_hz" in f else np.nan)
        mags = pd.to_numeric(f["mpsas"], errors="coerce")
        zero_flags.append(bool((mags == 0).any()))
    matched["sqm_frequency_Hz"] = frequencies
    matched["zero_magnitude_flag"] = zero_flags

    lv_spec, le_int, shape = [], [], []
    v_grid = np.interp(wl, cie_wl, cie_v, left=0, right=0)
    normalized = {}
    for intensity in matched.intensity.astype(int):
        Llam = spectra[intensity] / math.pi
        lv_spec.append(683.0 * mod.trapz(Llam*v_grid, wl))
        mask = (wl >= 360) & (wl <= 830)
        le_int.append(mod.trapz(Llam[mask], wl[mask]))
        area = mod.trapz(Llam, wl)
        normalized[intensity] = Llam/area
    ref = normalized[int(matched.intensity.iloc[0])]
    for intensity in matched.intensity.astype(int):
        cur = normalized[intensity]
        shape.append(100*np.sqrt(np.mean((cur-ref)**2))/np.max(ref))

    matched["spectral_luminance_cd_m2"] = lv_spec
    matched["integral_sensor_luminance_cd_m2"] = matched.illuminance_lx/math.pi
    matched["spectral_over_integral_luminance"] = matched.spectral_luminance_cd_m2/matched.integral_sensor_luminance_cd_m2
    matched["luminance_difference_percent"] = 100*(matched.spectral_over_integral_luminance-1)
    matched["integrated_radiance_360_830_W_m2_sr"] = le_int
    matched["spectral_shape_rms_change_percent"] = shape
    matched["L0_from_spectrum_cd_m2"] = matched.spectral_luminance_cd_m2*10**(0.4*matched.sqm_mpsas)
    matched["L0_from_integral_sensor_cd_m2"] = matched.integral_sensor_luminance_cd_m2*10**(0.4*matched.sqm_mpsas)
    matched["L0_spectrum_over_literature"] = matched.L0_from_spectrum_cd_m2/args.literature_L0
    matched["magnitude_predicted_literature"] = -2.5*np.log10(matched.spectral_luminance_cd_m2/args.literature_L0)
    matched["magnitude_offset_from_literature"] = matched.sqm_mpsas-matched.magnitude_predicted_literature

    # Fixed slope and per-level residual diagnostics.
    a = float(np.mean(matched.sqm_mpsas + 2.5*np.log10(matched.spectral_luminance_cd_m2)))
    L0_fit = 10**(a/2.5)
    matched["magnitude_predicted_fitted_L0"] = a-2.5*np.log10(matched.spectral_luminance_cd_m2)
    matched["fixed_slope_residual_mag"] = matched.sqm_mpsas-matched.magnitude_predicted_fitted_L0
    free = free_fit(matched.spectral_luminance_cd_m2, matched.sqm_mpsas)
    residual_level_r = correlation(matched.fixed_slope_residual_mag, np.log10(matched.spectral_luminance_cd_m2))
    residual_cct_r = correlation(matched.fixed_slope_residual_mag, matched.cct_K)

    # Frequency linearity.
    ok = np.isfinite(matched.sqm_frequency_Hz) & (matched.sqm_frequency_Hz > 0)
    freq_r2 = np.nan
    if ok.sum() >= 3:
        x, y = matched.loc[ok,"spectral_luminance_cd_m2"], matched.loc[ok,"sqm_frequency_Hz"]
        coef = np.polyfit(x, y, 1); pred=np.polyval(coef,x)
        freq_r2=float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2))

    max_lum_error=float(np.max(np.abs(matched.luminance_difference_percent)))
    l0_cv=float(100*np.std(matched.L0_from_spectrum_cd_m2,ddof=1)/np.mean(matched.L0_from_spectrum_cd_m2))
    checks=[]
    def add(name,status,value,meaning): checks.append({"check":name,"status":status,"value":value,"meaning":meaning})
    add("Spectrum vs BTS integral sensor/pi", "PASS" if max_lum_error<=args.tolerance_percent else "WARNING",
        f"max difference {max_lum_error:.3g}%", "Checks spectral units, selected BTS column, V(lambda) integration and E=pi L assumption.")
    add("Per-level L0 stability", "PASS" if l0_cv<=5 else "WARNING", f"CV {l0_cv:.3g}%",
        "A trend or large spread indicates nonlinearity, changing spectral mismatch or geometry drift.")
    add("Free fitted slope", "PASS" if np.isfinite(free['slope']) and abs(free['slope']+2.5)<=0.1 else "WARNING",
        f"slope {free['slope']:.6g}, R2 {free['r2']:.6g}", "The free slope should be near -2.5 if the constrained magnitude law is adequate.")
    add("Residual versus luminance", "PASS" if not np.isfinite(residual_level_r) or abs(residual_level_r)<0.7 else "WARNING",
        f"Pearson r {residual_level_r:.4g}", "A strong trend suggests nonlinearity or saturation.")
    add("Residual versus CCT", "PASS" if not np.isfinite(residual_cct_r) or abs(residual_cct_r)<0.7 else "WARNING",
        f"Pearson r {residual_cct_r:.4g}", "A strong trend suggests source-spectrum dependence.")
    add("Frequency versus luminance", "PASS" if np.isfinite(freq_r2) and freq_r2>=0.99 else "WARNING",
        f"R2 {freq_r2:.6g}", "Checks SQM raw-signal linearity; spectral changes may also affect it.")
    add("0.00 mpsas saturation", "WARNING" if any(zero_flags) else "PASS", str(any(zero_flags)),
        "Zero-magnitude readings should be excluded as possible upper-limit/saturation values.")
    add("Experimental versus literature L0", "INFO", f"fit/literature {L0_fit/args.literature_L0:.5g}",
        "Interpret only after the independent luminance check passes; SQM and CIE photopic passbands differ.")

    matched.to_csv(args.output/"diagnostic_level_results.csv",index=False)
    pd.DataFrame(checks).to_csv(args.output/"diagnostic_checks.csv",index=False)
    report={"selected_intensities":matched.intensity.astype(int).tolist(),"fitted_L0_cd_m2":L0_fit,
            "literature_L0_cd_m2":args.literature_L0,"fitted_over_literature":L0_fit/args.literature_L0,
            "free_fit":free,"L0_coefficient_of_variation_percent":l0_cv,
            "spectrum_integral_max_difference_percent":max_lum_error,
            "fixed_residual_vs_log_luminance_r":residual_level_r,"fixed_residual_vs_CCT_r":residual_cct_r,
            "frequency_vs_luminance_R2":freq_r2,"checks":checks}
    (args.output/"diagnostic_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (args.output/"diagnostic_report.txt").open("w",encoding="utf-8") as f:
        f.write("SQM CHARACTERISATION DIAGNOSTIC REPORT\n"+"="*42+"\n\n")
        f.write(f"Fitted L0: {L0_fit:.8e} cd m^-2\nLiterature L0: {args.literature_L0:.8e} cd m^-2\n\n")
        for c in checks: f.write(f"[{c['status']}] {c['check']}\n  {c['value']}\n  {c['meaning']}\n\n")

    # Figures.
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    x=matched.integral_sensor_luminance_cd_m2; y=matched.spectral_luminance_cd_m2
    lo,hi=min(x.min(),y.min()),max(x.max(),y.max())
    ax[0].scatter(x,y); ax[0].plot([lo,hi],[lo,hi],'--',label='1:1'); ax[0].set(xscale='log',yscale='log',xlabel='BTS integral sensor / pi (cd m$^{-2}$)',ylabel='Spectral luminance (cd m$^{-2}$)',title='Independent luminance check'); ax[0].legend()
    ax[1].axhline(args.literature_L0,ls='--',label=f'Literature {args.literature_L0:.3e}')
    ax[1].plot(matched.intensity,matched.L0_from_spectrum_cd_m2,'o-',label='Spectrum')
    ax[1].plot(matched.intensity,matched.L0_from_integral_sensor_cd_m2,'s--',label='Integral sensor / pi')
    ax[1].set(yscale='log',xlabel='Intensity',ylabel='$L_0$ (cd m$^{-2}$)',title='Per-level zero point'); ax[1].legend()
    for aax in ax:aax.grid(True,which='both',alpha=.3)
    fig.tight_layout();fig.savefig(args.output/'01_luminance_and_L0_checks.png',dpi=220);plt.close(fig)

    fig,ax=plt.subplots(1,3,figsize=(16,5))
    ax[0].axhline(0,color='black');ax[0].scatter(matched.spectral_luminance_cd_m2,matched.fixed_slope_residual_mag);ax[0].set_xscale('log');ax[0].set(xlabel='$L_v$ (cd m$^{-2}$)',ylabel='Residual (mag)',title='Residual vs luminance')
    ax[1].axhline(0,color='black');ax[1].scatter(matched.cct_K,matched.fixed_slope_residual_mag);ax[1].set(xlabel='CCT (K)',ylabel='Residual (mag)',title='Residual vs CCT')
    ax[2].plot(matched.intensity,matched.spectral_shape_rms_change_percent,'o-');ax[2].set(xlabel='Intensity',ylabel='RMS shape change (%)',title='Spectral-shape variation')
    for aax in ax:aax.grid(True,alpha=.3)
    fig.tight_layout();fig.savefig(args.output/'02_residual_and_spectral_checks.png',dpi=220);plt.close(fig)

    print(f"Fitted L0 = {L0_fit:.8e} cd m^-2")
    print(f"Fitted/literature = {L0_fit/args.literature_L0:.6g}")
    for c in checks: print(f"[{c['status']}] {c['check']}: {c['value']}")
    print("Saved checks to",args.output.resolve())
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}",file=sys.stderr); raise
