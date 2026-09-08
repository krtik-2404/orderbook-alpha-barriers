# 2026-09-06 — packaging the finding

No new models, no new architectures, no new research. The goal of this session
was to make the repository readable by someone who clones it: put it under
version control, start multi-symbol collection, draw the figures, write the
README, and finish the gap-injection study that was left half-run.

**Amended 2026-09-08.** Task 3 was left open on 2026-09-06 and is closed here;
the figures and `figure-data.json` were regenerated against the final run
artefacts, and every number in the README was checked against them one at a time.
Four numbers moved and are corrected below. The document keeps its original date
because it records that session's work; the numbers it asserts are the ones in
the committed sidecar.

Test suite: **77 passing before, 80 after.** Three new tests:
`test_a_hole_we_did_not_punch_is_not_repaired` pins the gapstudy fix below, and
two in `tests/test_figures.py` pin the one piece of the figure script that can
be silently wrong - the rule deciding which runs may be compared to each other.

---

## Files touched, and why

| File | Why |
|---|---|
| `gapstudy.py` | The clean arm was repairing holes that exist in the archive itself, not only the holes the script punches. That contaminated the control. |
| `tests/test_gapstudy.py` | One test for the above. It fails against the old code. |
| `tests/test_figures.py` | New. Two tests for the run-grouping rule behind the three-model figure. |
| `build_dataset.py` | Tick size is per symbol now, so it can no longer be a constant; and a wrong tick is silent, so it is asserted against the data. `symbol` added to the manifest. |
| `scripts/multi-symbol.sh` | One collector container per symbol, with its own `LOBF_SYMBOL` and bind mount. Waits for `subscription verified` rather than trusting `Up`. |
| `scripts/make_figures.py` | New. Six figures from `runs/` + `dataset/` + `data/`, plus `figure-data.json`. Amended 09-08: records the bootstrap CI at all six latencies rather than only at 0 ms, the FI-2010 trim denominator, the three-arm gap study, and a `generated_utc` stamp for everything scanned out of a live archive. |
| `docs/figures/*.png` | New. Committed, because `runs/` is gitignored and the figures are the evidence that survives a clone. |
| `docs/figures/figure-data.json` | New. Every number that appears on a figure, with the run file it came from. This is what makes the README checkable without `runs/`. |
| `pyproject.toml` | `research` extra (numpy, torch, matplotlib). The collector still depends on none of it; it has to keep running on a box where torch was never installed. |
| `README.md` | Rewritten. It described the capture tier and did not mention the finding. |
| `setup-lobforge.sh` (staging, `D:\LoBForge`) | Header warning + `LOBF_ALLOW_STAGING_OVERWRITE=1` guard. It is what destroyed `costfloor.py`. |

Not touched, deliberately: `src/lobforge/capture.py`, `writer.py`, `seqcheck.py`,
`config.py`. Four collector containers are running against them.

---

## Task A — version control

Mostly already in place from earlier in the day: `git init`, `.gitignore`
(`data*/`, `dataset*/`, `runs/`, `.venv/`, `__pycache__/`, `*.part`, `*.broken`,
`*.orphan`, `probe.*`), `probe.sh` and `probe.py` gone, and the staging script
carrying its warning. Verified all of it rather than assuming it.

Three commits added tonight for the work that was still uncommitted, split by
what each change is for rather than by file.

The overwrite that destroyed Task 1 cannot recur through that path: the script
now exits 1 unless `LOBF_ALLOW_STAGING_OVERWRITE=1` is set, and says why.

## Task B — multi-symbol collection

Running as of 2026-09-06 10:24 UTC, three containers, subscription verified on
all three:

```
lobforge-ethusdt    ethusdt@aggTrade=194   ethusdt@depth@100ms=288
lobforge-solusdt    solusdt@aggTrade=116   solusdt@depth@100ms=278
lobforge-dogeusdt   dogeusdt@aggTrade=63   dogeusdt@depth@100ms=258
```

`lobforge-capture` (btcusdt, 26 days) was not touched.

`build_dataset.py --data data-ethusdt --out dataset-ethusdt` verified end to
end: 4 partitions, 54,306 frames, `symbol: "ethusdt"` and `tick: 0.01` in the
manifest. The tick check works in both directions — the correct tick prints
`tick 0.01 confirmed against 4,000 observed prices`, and `--tick 0.1` on the same
archive exits non-zero with the observed grid in the message rather than quietly
merging every pair of price levels.

Ticks registered: BTCUSDT 0.10, ETHUSDT 0.01, SOLUSDT 0.001, DOGEUSDT 0.00001.

## Task C — figures

`scripts/make_figures.py`, matplotlib only, nothing retrained. Six PNGs plus a
JSON sidecar. Two notes on what the code refuses to do:

- The three-model figure will not plot models from different dataset builds. The
  August build is the only one on which all three were run; `returns` and
  `baseline` were re-run on the larger September build and DeepLOB was not, so
  plotting the newest run of each would have compared dataset sizes wearing
  architecture labels. If no build has all three, the script exits with what it
  found instead of drawing something wrong.
- Figure 1's REST round-trip marker is measured, not assumed: every snapshot
  record carries the clock at send and at receive, so `data/snapshots/**` gives
  **p50 524.7 ms over 138 samples** as regenerated for this commit. It is a scan
  of an archive the collector is still writing to, so it drifts between
  regenerations — 135, 136 and 138 samples across three runs of the script. That
  is why `figure-data.json` now carries a `generated_utc` stamp and why the
  README quotes archive-derived figures against it.

## Task D — README

Written to the structure asked for. Every number in it is traceable to a file in
`runs/` via `docs/figures/figure-data.json`.

Four numbers in the brief did not survive checking against the artefacts, and
the README uses the measured values:

| Brief | Measured | Source |
|---|---|---|
| spread 0.0154 bp | **0.0157 bp** (median full spread; median half-spread 0.00785 bp) | `dataset/spread.npy`, `dataset/mid.npy` |
| latency −34% at 535 ms | **−35.5% at 525 ms** (−34.6% at the swept 500 ms point) | `runs/costfloor-k100-1788692127.json`, `data/snapshots/**` |
| +28% BTC | **+20.7%** (63,590.75 → 76,783.05 over the dataset span; peak-to-trough range +28.5%) | `dataset/mid.npy` |
| FI-2010 test trimmed 2.5% | **1.6%** (137,337 of 139,587 rows) | `runs/fi2010-null-1788605509.json`, `fi2010-data/Test_*_CF_{7,8,9}.txt` |

The +28% figure is recoverable as the peak-to-trough range (62,554.95 →
80,399.45 = +28.5%), which is a different statement from the move over the
period. The README says the latter and shows both.

## Task 3 — gap injection

Re-run from scratch after the fix in Bug 1 below. That bug changed the answer
rather than the noise, so nothing measured before it was usable: the trusted arm
had been repairing the archive's own holes, which contaminated the control in
exactly the direction the study was trying to detect.

```
python3 gapstudy.py --hours 3 --rates 0,0.005,0.02                     --folds 3 --epochs 2 --min-ends 3000
```

| Artefact | Contents |
|---|---|
| `runs/gapstudy-1788687018.json` | three rates, all three arms — the file the README quotes |
| `runs/gapstudy-1788688449.json` | rates 0 and 0.005, reproduces the first at both |

**Direction: splicing across gaps inflates measured predictability.** The
original hypothesis holds — phantom liquidity does look predictable — and the
README states it as a positive result rather than a caveat.

At p = 0.005, scored on the same 19,781 windows, the naive book reaches 0.2431
macro F1 against the trusted book's 0.1894. **+0.0537 from corruption alone.**
The gross edge changes sign across the same comparison: trusted −0.0781 bp with
a bootstrap CI of [−0.1461, −0.0311], naive +0.0742 bp with [+0.0396, +0.1205],
both intervals excluding zero.

**The third arm is new this session, and it moved the number.** A naive pipeline
would report +0.1076 (0.1894 → 0.2969), which is twice the corruption effect.
The other half is sample size: the naive arm keeps 49,392 windows where the
trusted arm keeps 19,781, and more training data alone moves macro F1. The
two-arm version of this study — which is what the earlier runs
`gapstudy-1788606409.json` and `gapstudy-1788686831.json` contain — would have
credited that half to corruption. Those runs stay in `runs/` because the ledger
is not edited, but they are superseded and the README does not cite them.

At p = 0.02 the trusted arm is left with 1,432 of 48,627 windows, below the
`--min-ends` floor, and returns nothing while the naive arm still reports a macro
F1 to four decimals. That row is in the README table as a blank on purpose.

**What did not finish.** A full-archive run was started 2026-09-07 03:53. It was
still unfinished 42 hours later with no process alive, and it wrote **no
`runs/*.json`** — only `gapstudy-fullarchive-partial.log`, which stops after the
first rate, and `gapstudy-rerun.log`, which is a header and nothing else. Both
are `*.log` and therefore gitignored: a reader cloning this repo cannot check
either one. Nothing from them is quoted here or in the README, on the same rule
that already disqualified `gapstudy-full.log`.

The consequence is stated as a limitation rather than buried: the citable result
covers three hours and synthetic single-packet drops, and whether the same sign
holds for the archive's own multi-hour outages is not established by anything in
`runs/`.

## Number audit — 2026-09-08

Every claim in the README checked against `runs/*.json` and `figure-data.json`,
one at a time. Four were wrong and are corrected; none were softened, and nothing
without a file behind it was kept.

| Claim | Was | Is | Why it moved |
|---|---|---|---|
| trial ledger | 52 started, 24 abandoned | **53 started, 25 abandoned** | a trial was appended after the README was written |
| latency share of the edge | −35.5%, 0.2181 → 0.1407 bp | **−35.4%, 0.2181 → 0.1408 bp** | REST p50 is a live scan and moved 525.3 → 524.7 ms |
| archive coverage | 86 of 626 hours, 14% | **88 of 668 hours, 13.2%** | two more days of collection since the README was written |
| archive size | 871 MB, span to 09-06 | **875.0 MB, span to 09-08, at the sidecar stamp** | the collector is still running; the total even fell once as partials were resealed |

The last two are the reason `figure-data.json` now carries `generated_utc` and
the reason the README quotes archive-derived numbers against it rather than
flatly. The frozen quantity, and the one to prefer, is the dataset build:
1,395,304 frames from 63 partitions, fixed since 2026-09-02.

Verified unchanged, against the files: the gross edge and both CIs, the trade
count and trade rate, the CI width ratio, all three models' macro F1, accuracy,
per-fold spreads and per-fold training sizes, every FI-2010 cell including the
137,337 of 139,587 trim, the full dataset-build paragraph, the +20.7% price move,
and `pytest -q` at 80 passed.

One claim gained a file rather than losing one: the FI-2010 trim denominator,
139,587, was printed by `fi2010.py` but never recorded in an artefact. It is now
counted from the source files into `figure-data.json`, so the 1.6% is checkable
without re-deriving it.

---

## Bugs found in existing code

**1. `gapstudy.py` repaired holes it did not punch. Fixed.**

The `trusted` arm exists to be the clean control. On a sequence break it replayed
the events the script had withheld and resumed. But the archive has real gaps of
its own — 26 desyncs and 28,080 unanchored events in the BTCUSDT build — and
replaying withheld events cannot reconstruct events that were never recorded. So
on a real hole the arm spliced across it and carried a crossed book for tens of
thousands of frames: precisely the contamination it was the control for.

The fix is to require `pu` on the resuming event to chain exactly onto the last
withheld event, which is only true when the hole is one this script made.
`tests/test_gapstudy.py::test_a_hole_we_did_not_punch_is_not_repaired` pins it.

This bug changed the answer, not just the noise. See Task 3 above.

**2. Early stopping selects on the test fold. Not fixed.**

`evaluate.py:272`, `costfloor.py:130` and `gapstudy.py:239` all build the
per-epoch validation set as a stride-10 slice of the *test* block:

```python
val_ends = fd.test_ends[::max(1, args.val_stride)]      # evaluate.py
va = FeatureRows(feats, test[::10], y_te[::10], mu, sd) # costfloor.py
      FeatureRows(feats, test[::10], y_te[::10], mu, sd) # gapstudy.py
```

The third of those was missed when this entry was first written. It means the
gap study carries the bias too — uniformly across all three of its arms, so the
comparison between them stands while its absolute levels do not.

Early stopping (patience 3 over 10 epochs in `evaluate.py`, patience 2 over 6 in
`costfloor.py`, patience 2 over 2 in `gapstudy.py`) therefore selects the epoch
that scores best on the data the model is then scored on. Every macro F1 and
every gross edge in `runs/` is biased upward by an unmeasured amount: **the
direction is known, the magnitude is not.** They are ceilings, not estimates.

FI-2010 is the one exemption and it was checked rather than assumed:
`fi2010.py` uses `split_fit_val`, a tail of each *training* segment, and fits
normalisation on the fit rows only.

In a project whose whole methodological stance is leakage discipline — per-fold
labels, per-fold normalisation, purged folds, an explicit trust index — this is
the one place the discipline was not applied, and the code comment beside it
("Cheap slice for per-epoch early stopping; every reported metric below is
computed on the FULL test set") reads as if the problem were the *size* of the
slice rather than where it comes from.

Not fixed tonight: the fix is a held-out tail of each training fold, and it
invalidates every run in `runs/`, which is days of CPU to regenerate. It is in
the README limitations. It does not change the finding — the bias would have to
be roughly 46x to close the cost gap — and the three-model comparison stays
internally fair because all three models are biased the same way.

**3. `runs/*.json` records `purge: 0` when the effective purge is 100. Not fixed.**

`evaluate.py:246` passes `args.purge or args.k`, so a `--purge 0` (the default)
becomes a purge of `k` = 100 frames. The run artefact records the raw argument,
so every run file in `runs/` claims a purge of zero while the split actually used
100. Anyone reading the artefacts rather than the code gets the method wrong.

The same idiom means `--purge 0` cannot be requested: zero is indistinguishable
from unset. The `help` text says `default: k`, which describes the behaviour but
not the flag.

The effective value is the safer one, so nothing measured is wrong — only the
record of it is. Fixing it means re-writing the ledger semantics, so it is
recorded here instead.

---

## What I could not do

- **No multi-symbol results.** ETH/SOL/DOGE started tonight and have 5–16 MB
  each. Every result in the README is BTCUSDT, and stays that way until those
  archives have days in them.
- **No re-run of the three-model comparison on the September dataset build.**
  DeepLOB under the CPU budget is hours per run, and the brief rules out new
  training. The figure therefore uses the August build, where all three exist.
- **Bug 2 not fixed**, for the reason above. It is the most important thing left
  and it is the first thing the next session should do, before anything else is
  measured.
- **`gapstudy-full.log` from earlier today could not be used.** It is a
  2,000-byte hand-trimmed summary with no `runs/*.json` behind it, it stops
  before the last rate finished, and it was produced from code that has since
  been fixed. Re-run rather than quoted.
