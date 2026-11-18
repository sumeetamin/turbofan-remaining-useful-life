# Turbofan Remaining Useful Life

A reproducible NASA C-MAPSS prognostics benchmark with engine-disjoint evaluation and a fixed 100-cycle censor point.

## Dataset and run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py
python benchmark.py
```

The fetcher retrieves NASA's C-MAPSS archive and extracts the FD001 run-to-failure training file. Source and row-level data remain local and are excluded from Git.

## Evaluation

The model is trained on RUL labels capped at 125 cycles. At the fixed cycle-100 evaluation, it reports both capped-label MAE and raw-target MAE: the current split gives 15.77 cycles capped and 25.53 cycles raw. Nine of 25 held-out engines had raw RUL above 125. The raw-target result scores capped-model predictions against uncapped truth; it is not an uncapped-trained model. All earlier measurements for training engines are used to fit a Random Forest with trailing 10-cycle sensor statistics. Test engines have no overlap with training engines. The report compares against the training engines' median RUL at cycle 100 and includes an asymmetric NASA-style score.

See `reports/benchmark.json` for the measured results and SHA-256 fingerprints.

## Limits

C-MAPSS is simulator-generated, not observed commercial fleet data. FD001 has one simulated operating condition and one simulated fault mode. Holding out engines within FD001 does not establish performance on other operating conditions, faults, engines, or real fleets. The fixed cycle-100 evaluation does not establish an alert or maintenance policy. Do not use these results for aviation maintenance decisions without validation on authorized fleet data.

## Source

A. Saxena and K. Goebel (2008), “Turbofan Engine Degradation Simulation Data Set,” NASA Ames Prognostics Data Repository. NASA describes the trajectories as generated using the C-MAPSS simulation. [NASA PCoE data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

No code license file is included yet. Dataset terms are governed by NASA's source.

