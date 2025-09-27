# Assignment 7 - CPA on T-Table Based AES (FPGA) - Ready Package

**Deliverables produced:** `key.txt`, 16×`plot_k_corr_*`, 16×`plot_nrtraces_corr_*`, 16×`plot_points_corr_*`, `description.txt`, plus source code.

**How to run**
```
pip install -r requirements.txt
python run_assignment7.py
```
(Adjust paths or call `cpa_assignment7.py` directly if needed.)

**Models:** `sbox_hw` (default), or `tword_hw` / `tbyte{0..3}_hw` for T-table leakage.  
**ROI:** top-variance points; tune with `--roi` (e.g., 800..2000).  
**Steps:** number of prefixes for the nrtraces plot (default 20).


