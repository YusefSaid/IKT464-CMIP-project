# Future Marine Conditions under Climate Change

IKT464 Advanced ICT Project, University of Agder, autumn 2026.

How does the global wave climate of the CMIP6 model **EC-Earth3** (simulated with the wave model WAVEWATCH III) change at the end of the century under a low-emission (**SSP1-2.6**) and a high-emission (**SSP5-8.5**) pathway? The code reduces three 30-year, 3-hourly data files (about 60 GB each, 142,868 sea points) to statistics per sea point and compares the future (2071–2098) with the historical period (1985–2014).

**Main findings (SSP5-8.5):** mean significant wave height increases in the Southern Ocean (+0.23 m, +6 %) and decreases in the North Pacific (−0.20 m) and North Atlantic (−0.15 m); the global mean hardly changes. The change exceeds natural year-to-year variability over 43 % of the ocean (SSP1-2.6: 6 %). The wind changes in the same pattern, and the sea-ice-affected area shrinks strongly.

## Repository structure

```
├── src/                         scripts (run from the project root)
│   ├── download_data.py         resumable download of the data files
│   ├── check_downloads.sh       status of running downloads
│   ├── scan_chunks.py           integrity check: decompresses every data block
│   ├── repair_chunks.py         re-downloads only damaged parts
│   ├── compute_stats.py         statistics per sea point (main computation)
│   ├── cmip_utils.py            shared helpers: reading files, grid, maps, saving
│   ├── report_style.py          figure style at the printed size of the report
│   └── make_report_numbers.py   numbers and regional table for the report
├── notebooks/
│   ├── 00_data_overview.ipynb   what the data contain: files, periods, variables, grid, missing values
│   ├── 01_change_signal.ipynb   results: change maps, significance, sea ice, regional table
│   └── archive/                 early exploration (first 10 years, partial downloads)
├── fig/                         figures (PNG and PDF): change/, overview/, eda/ (early exploration)
├── results/integrity/           integrity check log and list of damaged data blocks
├── data/                        data files and computed statistics (not in the repository)
└── requirements.txt
```

## Getting started

**1. Environment** (Python 3.12)

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**2. Data** (not included; provided by the MOREnergy Lab, Politecnico di Torino). Place the files as:

```
data/raw/historical/EC-EARTH3.mat
data/raw/ssp126/EC-EARTH3.mat
data/raw/ssp585/EC-EARTH3.mat
data/raw/SeaMask_WW3.mat
```

**3. Check the files** (lists damaged data blocks, if any, in `data/raw/<experiment>/bad_chunks.csv`)

```bash
.venv/bin/python src/scan_chunks.py historical ssp126 ssp585
```

**4. Compute the statistics** (about 4 minutes per file and variable on 5 cores; written to `data/processed/stats30/`)

```bash
.venv/bin/python src/compute_stats.py --scenarios historical --vars hs wspd tp
.venv/bin/python src/compute_stats.py --scenarios ssp126 ssp585 --vars hs wspd tp --last-year 2098
```

**5. Run the notebooks:** `00_data_overview`, then `01_change_signal`. In the setup cell of `01_change_signal`, `VAR` selects the variable (`"hs"` wave height, `"wspd"` wind speed, `"tp"` peak period); use Restart and Run All after changing it. Figures go to `fig/`, tables to `data/processed/`.

**6. Numbers for the report** (optional): `.venv/bin/python src/make_report_numbers.py` writes the key numbers and the regional table as LaTeX files.

## Notes

- **Scenario period 2071–2098.** The last two years of both scenario files were damaged during the transfer and could not be repaired; `--last-year 2098` keeps the analysis on intact data. The evidence is in `results/integrity/`.
- **Missing values** occur only in ice-covered polar seas (the wave model computes no waves under sea ice). Points missing more than 10 % of the time are masked in all comparisons.
- **Memory:** the files are read in blocks of 330 sea points, so a full run needs only a few GB of memory, although one variable is 47 GB uncompressed.
