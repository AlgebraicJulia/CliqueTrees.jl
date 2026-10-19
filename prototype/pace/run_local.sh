#!/bin/bash
# Local run: 30 s per instance, at most 4 solver processes at a time. Needs tools/ built (see README).
cd "$(dirname "$0")"; T=${T:-30}
for s in tamaki17 septw; do for set in exact bonus; do python3 -I tools/run.py $s $set $T 4 results; done; done
python3 -I tools/run.py tamaki16 exact $T 4 results
python3 -I tools/run.py uplow22 exact $T 2 results   # 2 threads each
