#!/usr/bin/env bash
# Sanity-check that the datasets referenced by the configs are in place.
set -uo pipefail
ok=0
check() { if [ -e "$1" ]; then echo "  [ok]      $1"; else echo "  [missing] $1"; ok=1; fi; }
echo "CheXpert:"
check data/CheXpert-v1.0-small/train.csv
check data/CheXpert-v1.0-small/valid.csv
echo "NIH ChestX-ray14:"
check data/nih/Data_Entry_2017.csv
check data/nih/images
exit $ok
