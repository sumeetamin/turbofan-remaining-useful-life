# Turbofan Remaining Useful Life

An engine-group holdout RUL regression baseline with rolling sensor features, capped targets, and saved predictions.

> **Portfolio prototype.** Code is provided; no benchmark results are claimed until you run it on the cited data and record the output.

## Use case

An engine-group holdout RUL regression baseline with rolling sensor features, capped targets, and saved predictions. The command below reads a local data export, applies a transparent baseline, writes holdout or decision-support outputs, and saves the fitted model where applicable.

## Data and provenance

[NASA C-MAPSS engine simulator data](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data). A local C-MAPSS run-to-failure training file such as `train_FD001.txt`; the repository does not redistribute NASA data. Dataset files are not bundled. Respect the source terms and cite the dataset when publishing results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download/prepare the source data locally, then run:
python main.py --train data/train_FD001.txt --out outputs
```

## Outputs

The `outputs/` directory contains the model-specific results (metrics and/or ranked CSV) and a serialized estimator where applicable. Outputs are reproducible with the same input, package versions, and random seed. The repository intentionally excludes downloaded data and generated artifacts.

## Evaluation and operating notes

C-MAPSS is simulated, not fleet evidence. NASA currently notes access restrictions on its landing page; check the official source before downloading. This model must not be used for maintenance decisions without validation on real fleet data.

For a production review, preserve a time-based, untouched holdout; compare against a simple business baseline; inspect calibration and subgroup error; document data lineage; and monitor drift. The included baseline is not a production service.

## License

No code license file is included yet. Third-party datasets remain subject to their original terms.
