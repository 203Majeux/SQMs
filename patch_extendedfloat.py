#!/usr/bin/env python3
"""Patch run2.py to avoid eager ExtendedFloat decoding."""
from pathlib import Path

source = Path("run2.py")
if not source.exists():
    raise SystemExit(f"File not found: {source.resolve()}")

text = source.read_text(encoding="utf-8")
text = text.replace("from nptdms import types as tdms_types\n", "")
text = text.replace(
    "\n# npTDMS workaround for unrelated ExtendedFloat channels.\n"
    "tdms_types.ExtendedFloat.size = 16\n",
    "\n",
)
text = text.replace(
    "    tdms_file = TdmsFile.read(tdms_path)\n",
    "    # Open lazily so unrelated ExtendedFloat channels are not decoded.\n"
    "    tdms_file = TdmsFile.open(tdms_path)\n",
)

needle = """    for idx in reference_ids:
        reference_net_by_id[idx] = (
            read_float(measurements, signal_channels[idx], n)
            - read_float(measurements, dark_channels[idx], n)
        )

    # Average only finite positive reference currents point by point.
"""
replacement = """    for idx in reference_ids:
        reference_net_by_id[idx] = (
            read_float(measurements, signal_channels[idx], n)
            - read_float(measurements, dark_channels[idx], n)
        )

    # All required TDMS arrays are now in memory.
    tdms_file.close()

    # Average only finite positive reference currents point by point.
"""

if needle not in text:
    raise SystemExit("Could not find the reference-reading block. No output written.")
text = text.replace(needle, replacement)

output = Path("run_irradiance_f0_fixed.py")
output.write_text(text, encoding="utf-8")
compile(text, str(output), "exec")
print(f"Created: {output.resolve()}")
