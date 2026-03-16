#!/usr/bin/env python3
"""
Visualize ABM tutorial results.

Defines 3 parameter sets per tutorial (9 runs total). Pick which one to
execute by setting TUTORIAL and RUN below, then run this script. The
executable is called with the chosen parameters and the output is plotted.

Usage:
  python visualize_abm.py          # run the selected TUTORIAL / RUN
"""

import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import sys
import os

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION – pick ONE tutorial and ONE run (1, 2, or 3)
# ══════════════════════════════════════════════════════════════════════
TUTORIAL = "vaccination"       # "household", "testing", or "vaccination"
RUN = 4               # 1, 2, 3, ...  (see tables below)

# ── 3 runs per tutorial ──────────────────────────────────────────────
#
# Tutorial 1 – Households:  args = [n_households, infected_frac, sim_days]
#   Run 1: small  population, low  infection, short sim
#   Run 2: medium population, moderate infection, medium sim  (default)
#   Run 3: large  population, high infection, long sim
#
# Tutorial 2 – Testing:     args = [testing_prob, validity_days, n_households]
#   Run 1: no testing baseline
#   Run 2: moderate testing (default)
#   Run 3: aggressive testing, short validity
#
# Tutorial 3 – Vaccination: args = [vaccination_rate, n_households, protection_peak, use_data_vacc]
#   Run 1: no vaccination baseline
#   Run 2: moderate campaign vaccination
#   Run 3: high campaign vaccination, strong protection
#   Run 4: data-driven vaccination (from JSON), large population

RUNS = {
    "household": {
        1: {"args": ["50",  "0.05", "15"], "label": "small pop, low inf, 15d"},
        2: {"args": ["125", "0.2",  "30"], "label": "medium pop, moderate inf, 30d (baseline)"},
        3: {"args": ["500", "0.5",  "10"], "label": "large pop, high inf, 90d"},
    },
    "testing": {
        1: {"args": ["0.0", "3",  "125"], "label": "no testing (baseline)"},
        2: {"args": ["0.5", "3",  "125"], "label": "50% testing, 3d validity"},
        3: {"args": ["1.0", "1",  "125"], "label": "100% testing, 1d validity"},
    },
    "vaccination": {
        1: {"args": ["0.0",  "125", "0.67", "0"], "label": "no vaccination (baseline)"},
        2: {"args": ["0.3",  "125", "0.67", "0"], "label": "30% campaign"},
        3: {"args": ["0.7",  "125", "0.95", "0"], "label": "70% campaign, strong protection"},
        4: {"args": ["0.0",  "500", "0.67", "0"], "label": "no vaccination(JSON), 500 hh"},
        5: {"args": ["0.0",  "500", "0.67", "1"], "label": "data-driven (JSON), 500 hh"},
    },
}

# ── Executable / output-file mapping ─────────────────────────────────
_config = {
    "household":   {"exe": "tutorial_abm_household",   "file": "abm_household.txt",    "title": "Tutorial 1 – Households"},
    "testing":     {"exe": "tutorial_abm_tests",        "file": "abm_tests.txt",         "title": "Tutorial 2 – Testing"},
    "vaccination": {"exe": "tutorial_abm_vaccination",  "file": "abm_vaccination.txt",   "title": "Tutorial 3 – Vaccination"},
}

cfg = _config[TUTORIAL]
run_cfg = RUNS[TUTORIAL][RUN]
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
bin_dir = os.path.join(root_dir, "build", "bin")
exe_path = os.path.join(bin_dir, cfg["exe"])
fpath = os.path.join(bin_dir, cfg["file"])

# ── Run the executable ────────────────────────────────────────────────
cmd = [exe_path] + run_cfg["args"]
print(f"[{TUTORIAL} / run {RUN}] {run_cfg['label']}")
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=bin_dir)
if result.returncode != 0:
    print(f"ERROR: {cfg['exe']} exited with code {result.returncode}")
    sys.exit(1)

# ── Read results ──────────────────────────────────────────────────────
data = pd.read_csv(fpath, sep=r'\s+')

# ── Infection-state columns & colours ─────────────────────────────────
states = ['S', 'E', 'I_NS', 'I_Sy', 'I_Sev', 'I_Crit', 'R', 'D']
colors = ['blue', 'orange', 'gold', 'red',
          'darkred', 'purple', 'green', 'black']
labels = ['Susceptible', 'Exposed', 'Infected (No Symptoms)',
          'Infected (Symptomatic)', 'Infected (Severe)',
          'Infected (Critical)', 'Recovered', 'Dead']

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
for s, c, l in zip(states, colors, labels):
    ax.plot(data['Time'], data[s], label=l, color=c, linewidth=2)

final = data.iloc[-1][states]
n_agents = int(final.sum())

ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Number of Persons', fontsize=12)
ax.set_title(f'{cfg["title"]}  –  Run {RUN}: {run_cfg["label"]}  (N = {n_agents})',
             fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
outpng = f'abm_{TUTORIAL}_run{RUN}.png'
plt.savefig(f'abm_{TUTORIAL}_run{RUN}.png', dpi=300, bbox_inches='tight')
print(f"\nFinal counts:\n{final.to_string()}")
print(f"Plot saved at '{os.path.abspath(os.getcwd()) + '/' + outpng}'")
plt.show()
