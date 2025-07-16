#!/usr/bin/env python
# coding: utf-8

# # Calculate ORC WER
# Usage: python orc_wer_sim.py <path/to/eval/wer_dir>

from utils.evaluation import calc_wer, aggregate_wer_metrics
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm


METRICS = ["cp_wer", "tcorc_wer", "tcp_wer"]

exp_dir = Path(sys.argv[1])
for c in [5, 10, 15, 20, 25, 30]:
    dfs = []
    print(c, " ")
    for wer_dir in tqdm(list(exp_dir.glob("*"))):
        if wer_dir.name.endswith(".csv"):
            continue
        dfs.append(calc_wer(None, wer_dir/"tcp_wer_hyp.json", wer_dir/"tc_orc_wer_hyp.json", wer_dir/"ref.json", metrics_list=METRICS, collar=c))

    all_session_wer_df = pd.concat(dfs, ignore_index=True)
    metrics = aggregate_wer_metrics(all_session_wer_df, METRICS)
    print(metrics)
