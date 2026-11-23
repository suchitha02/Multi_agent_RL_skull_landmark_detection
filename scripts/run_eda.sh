#!/usr/bin/env bash
# Full exploratory data analysis for both datasets, packed into eda_results.zip to share.
#   bash scripts/run_eda.sh /path/to/ISBI2015 /path/to/Aariz
# ISBI2015 must contain RawImage (bmp images) and the 400_junior / 400_senior folders (any nesting).
# Aariz is the unzipped figshare folder (train / valid / test + cephalogram_machine_mappings.csv).
set -euo pipefail
ISBI_RAW=${1:?path to ISBI 2015 folder}
AARIZ_RAW=${2:?path to Aariz folder}

[ -f data/isbi/train/meta.json ]  || python scripts/prepare_data.py isbi  --raw "$ISBI_RAW"  --out data/isbi
[ -f data/aariz/train/meta.json ] || python scripts/prepare_data.py aariz --raw "$AARIZ_RAW" --out data/aariz

python scripts/analyze_dataset.py --data-root data/isbi  --out analysis/isbi
python scripts/analyze_dataset.py --data-root data/aariz --out analysis/aariz
python scripts/analyze_dataset.py --data-root data/isbi  --compare data/aariz --out analysis/isbi_to_aariz
python scripts/analyze_dataset.py --data-root data/aariz --compare data/isbi  --out analysis/aariz_to_isbi

python - <<'PY'
import shutil
shutil.make_archive("eda_results", "zip", "analysis")
print("-> eda_results.zip (send this back for review)")
PY
