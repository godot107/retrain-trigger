"""Build the notebooks, then execute them so outputs are embedded:

    .venv/bin/python notebooks/build.py          # all (or: build.py 01)
    for n in notebooks/0*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace $n; done

Needs `python -m retrain.experiment` and `python -m retrain.realdata` run first (results/).
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SETUP = """
%matplotlib inline
import os, json, warnings
os.chdir('..')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from retrain import plot
plot.style()
pd.set_option('display.width', 160)
LABELS = {'never': 'never retrain', 'schedule': 'every 10 batches', 'psi': 'input drift: PSI',
          'psi_rule': 'input drift: PSI > 0.2 rule', 'ks': 'input drift: KS test', 'c2st': 'input drift: classifier test',
          'pred_psi': 'prediction drift: PSI', 'error': 'error ratio', 'page_hinkley': 'Page-Hinkley (error)',
          'psi_error': 'input flag + error confirms', 'error_cp': 'error ratio + change point',
          'page_hinkley_cp': 'Page-Hinkley + change point', 'oracle': 'oracle (knows the change)'}
"""


def _nb():
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731
    return cells, md, code


def nb01():
    c, md, code = _nb()
    md("""
# 1 · Which signal should tell a data scientist to retrain?

**The situation:** a model is in production. The world it learned from may change. A monitoring rule
watches something and, when it crosses a line, pages a data scientist to retrain. Retraining costs
time and compute; *not* retraining costs bad predictions. What should the rule watch?

**The test bench:** simulated data streams (think: one batch per week, 500 rows each) where we decide
exactly when and how the world changes, so we know the right answer:

| Scenario | What changes | Should we retrain? |
|---|---|---|
| none | nothing | no |
| covariate_harmless | the inputs move, but where the model already works | no |
| covariate_harmful | the inputs move where the model never learned | yes |
| concept_sudden | the *meaning* changes: same inputs, different right answer | yes |
| concept_gradual | the same, phased in over 20 batches | yes |
| concept_complex | the relationship becomes much more intricate | yes |

The change starts at batch 40 of 80. The model (gradient-boosted trees, tuned at the start) trains on
batches 0-19. True answers ("labels") come back 2 batches late, as they do in real life.
""")
    code(SETUP + "\nfrom retrain import simulate, detectors\nd = pd.read_csv('results/synthetic_runs.csv')\nt = d[d.study == 'triggers']")
    md("""
## 1 · What the drift looks like

For three scenarios, the first input feature and the target, before (grey) and after (colour) the change.
In *concept_sudden* the inputs look identical: only the relationship between input and target moved.
""")
    code("""
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, sc in zip(axes, ['covariate_harmless', 'covariate_harmful', 'concept_sudden']):
    s = simulate.stream(sc, 0)
    for b, col in ((30, plot.MUTED), (55, plot.SERIES[0])):
        ax.scatter(s.X[b][:, 0], s.y[b], s=3, alpha=0.4, color=col, label='before' if b < 40 else 'after')
    ax.set_title(sc); ax.set_xlabel('input x0')
axes[0].set_ylabel('target'); axes[0].legend(markerscale=4)
plt.tight_layout()
""")
    md("""
## 2 · A fair comparison: same false-alarm rate for every trigger

A more sensitive trigger catches more and also cries wolf more. So every trigger's threshold is set so
that on *drift-free* streams (50 calibration streams, never reused for testing) it fires in about 1 batch
in 300. The common industry rule "PSI above 0.2" is kept as an extra, uncalibrated row.
""")
    code("pd.Series(json.load(open('results/thresholds.json'))['thresholds']).round(3).rename('calibrated threshold')")
    md("""
## 3 · The result: 30 streams per scenario

**Total cost** = summed prediction error over the 60 production batches (retrains not yet priced in).
Lower is better. ± = 95% interval over the 30 streams.
""")
    code("""
def ci(x):
    rng = np.random.default_rng(0); b = rng.choice(x.values, (2000, len(x))).mean(1)
    return (np.percentile(b, 97.5) - np.percentile(b, 2.5)) / 2
order = ['never', 'schedule', 'psi_rule', 'psi', 'ks', 'c2st', 'pred_psi', 'psi_error', 'error', 'page_hinkley',
         'error_cp', 'page_hinkley_cp', 'oracle']
g = t.groupby(['trigger', 'scenario']).error_total
tab = (g.mean().round(1).astype(str) + ' ± ' + g.apply(ci).round(1).astype(str)).unstack().reindex(order)
tab.index = [LABELS[i] for i in tab.index]
tab[list(simulate.SCENARIOS)]
""")
    code("""
harm = ['covariate_harmful', 'concept_sudden', 'concept_gradual', 'concept_complex']
h = t[t.scenario.isin(harm)]
summary = pd.DataFrame({
    'concept drift caught (share of streams)': h[h.scenario.str.startswith('concept')].assign(c=lambda x: x.first_retrain_delay.notna() & (x.first_retrain_delay < 15)).groupby('trigger').c.mean(),
    'median batches to first retrain': h.groupby('trigger').first_retrain_delay.median(),
    'wasted retrains per harmless stream': t[t.scenario == 'covariate_harmless'].groupby('trigger').wasted_retrains.mean(),
}).reindex(order).round(2)
summary.index = [LABELS[i] for i in summary.index]
summary
""")
    md("""
**Reading it:**

- **Input-drift triggers are blind to concept drift.** PSI, KS, the classifier test and prediction
  drift fire on the meaning-changes only at their false-alarm rate (the occasional "catch" 15+ batches
  later is luck). The PSI > 0.2 rule never fires at all.
- **They pay for harmless drift:** about 3 retrains per stream when the inputs moved somewhere the
  model already handled.
- **Error-based triggers catch every harmful change** within 2-5 batches, and almost never retrain for
  nothing. Their catch: they need labels.
- **Where input drift wins:** when inputs move into new territory, it fires instantly (before any labels
  arrive), and is slightly cheaper there (37.7 vs 40.6).

## 4 · Detecting fast isn't enough: retrain on the right data

The plain error trigger notices sudden concept drift within 2 batches, yet costs far more than the
oracle (62.9 vs 38.9). It retrains on the last 10 batches, which are mostly *from before the change*,
so the new model is a blend of the old world and the new one.

**Change-point window:** Page-Hinkley's running sum of excess error has its lowest point just before
the change began. Retraining only on data after that point (and waiting until there is enough of it)
nearly closes the gap: 43.0, with half the retrains. It hurts when inputs move to new territory (47.0
vs 40.6), where the post-change window is too small.

## 5 · Pricing the retrain: which trigger is cheapest?

Each retrain costs a data scientist's time and compute. Expressed in "batches of normal prediction
error", the cheapest trigger on average across all six scenarios:
""")
    code("""
cc = pd.read_csv('results/cost_curve_synthetic.csv', index_col=0)
cc.columns = cc.columns.astype(int)
show = ['never', 'schedule', 'psi', 'psi_error', 'error', 'page_hinkley', 'error_cp', 'page_hinkley_cp']
fig, ax = plt.subplots(figsize=(8, 4))
for k, name in enumerate(show):
    ax.plot(cc.columns, cc.loc[name], marker='o', markersize=4, color=plot.SERIES[k % 8], label=LABELS[name])
ax.set_xscale('symlog', linthresh=1); ax.set_xlabel('cost of one retrain (in batches of normal error)')
ax.set_ylabel('average total cost'); ax.set_title('Cheapest trigger depends on what a retrain costs')
ax.legend(fontsize=8, ncol=2); plt.tight_layout()
best = cc.drop(index='oracle').idxmin().map(LABELS)
best.rename('cheapest trigger').to_frame().T
""")
    md("""
- **Cheap retrains (0-2):** Page-Hinkley with a change-point window.
- **Moderate (5-10):** error ratio with a change-point window: it retrains less often.
- **Very expensive (20+):** in this simulation, never retraining wins: no change is costly enough
  to pay for the data scientist. In a real business that threshold is a number worth knowing.
- **Input-drift triggers are never the cheapest**, at any price.

**What this means in practice:** page on *performance* when labels come back reasonably fast; use input
drift as an early, low-priority flag (and as the only signal when labels never come back); and when you
retrain, train on data from after the change, not a fixed window.
""")
    return c


def nb02():
    c, md, code = _nb()
    md("""
# 2 · Retrain how? New weights, new hyperparameters, or keep training?

When the trigger fires, a retrain can:

- **refit:** same hyperparameters, new weights learned from recent data;
- **retune:** a fresh hyperparameter search on recent data, then fit ("reset weights and hyperparameters");
- **warm start:** keep the old trees and add more, fitted to recent data.

Same 30 streams per scenario as notebook 1; mean prediction error after the change (lower is better).
""")
    code(SETUP + "\nd = pd.read_csv('results/synthetic_runs.csv')")
    code("""
a = d[(d.study == 'actions') | ((d.study == 'triggers') & d.trigger.isin(['error', 'oracle', 'schedule']))]
tab = a.groupby(['scenario', 'trigger', 'action']).error_after_onset.mean().unstack().round(3)
tab[['refit', 'retune', 'warm']]
""")
    md("""
- **Retuning matters when the world gets more complicated.** In *concept_complex*, the settings tuned
  for the simpler world hold the model back; a fresh search cuts the error by 15-22% whichever trigger
  fired. Elsewhere retuning helps by only 1-6%: fresh data is what matters.
- **Warm start is the worst option, and dangerous when the inputs move.** After *covariate_harmful*,
  warm-started models are 4-5x worse than refitted ones.

## Why warm start fails here

The warm start only *adds* trees, so in principle it should at least match the old model. A small
experiment: train on inputs centred at 0, then warm-start on inputs shifted by 0, 1 or 4.
""")
    code("""
from sklearn.ensemble import HistGradientBoostingRegressor as H
rng = np.random.default_rng(0); rows = []
for shift in (0.0, 1.0, 4.0):
    X = rng.normal(0, 1, (3000, 1)); y = X[:, 0]
    m = H(max_iter=50, random_state=0).fit(X, y)
    X2 = rng.normal(shift, 1, (3000, 1)); y2 = X2[:, 0]
    m.set_params(warm_start=True, max_iter=150); m.fit(X2, y2)
    fresh = H(max_iter=150, random_state=0).fit(X2, y2)
    rows.append({'input shift': shift, 'warm start error': np.abs(m.predict(X2) - y2).mean(),
                 'fresh fit error': np.abs(fresh.predict(X2) - y2).mean()})
pd.DataFrame(rows).round(3)
""")
    md("""
The damage grows with the shift. Reading scikit-learn's source explains it: on each `fit`, this model
re-learns how to bucket ("bin") each feature from the *new* data, but while adding trees it evaluates the
*old* trees on the newly bucketed data, whose split points referred to the old buckets. The new trees
learn to correct a wrong starting point; at prediction time the old trees use their original real-valued
splits, so the correction no longer matches. **Practical rule: with this library, warm-start only on data
from the same distribution; after drift, refit.**

## How fast do labels need to come back?

The error-based trigger needs true answers. Same scenarios, labels returned 1 to 10 batches late:
""")
    code("""
dl = d[(d.study == 'delay') | ((d.study == 'triggers') & (d.trigger == 'error'))]
dl.groupby(['scenario', 'label_delay']).error_total.mean().unstack().round(1)
""")
    md("""
Slower labels cost more, roughly steadily, with one wrinkle: at a 5-batch delay the first retrain happens
later and so trains on *more* post-change data, which can beat a 2-batch delay. The same lesson as the
change-point window: what you retrain on matters as much as when.
""")
    return c


def nb03():
    c, md, code = _nb()
    md("""
# 3 · Real data: day-ahead electricity demand, 2019-2025

**The model:** for 12 large US grid operators, predict each hour of tomorrow's electricity demand from
the calendar and recent demand (yesterday, two days ago, last week). Trained on 2019, then used week by
week through 2025. Actual demand arrives the next day, so the true answers come back within a week.
Errors are a fraction of the operator's typical load (0.05 = 5%). No weather data: a deliberately
simple model, which is fine for comparing *retraining rules*.

**What changed in the real world:** COVID lockdowns (March 2020) reshaped daily demand; Winter Storms Uri
(Feb 2021) and Elliott (Dec 2022) were extreme; Texas (ERCOT) and Florida demand grew fast.

**Honest setup:** there is no drift-free stretch to calibrate on and no known "right" retrain date. So
each trigger's threshold is **chosen on 2020-2021** and **scored once on 2022-2025**.
""")
    code(SETUP + """
from retrain import realdata, eia, backtest
rows = json.load(open('results/real_runs.json'))
t = realdata.table(rows)
never = {r['ba']: r for r in rows if r['trigger'] == 'never'}
""")
    code("""
yr = pd.DataFrame({ba: pd.Series(r['mae'], index=pd.to_datetime(r['weeks'])).groupby(lambda w: w.year).mean()
                   for ba, r in never.items()}).T
(yr * 100).round(1).rename(columns=lambda c: f'{c} error %')
""")
    md("""
Without retraining, most operators drift slowly. **ERCOT's error goes from 5% to 9%**: demand grows past
anything seen in 2019, and tree models cannot predict values beyond their training range.
""")
    code("""
fig, ax = plt.subplots(figsize=(11, 3.6))
for k, ba in enumerate(['ERCO', 'FPL', 'PJM', 'CISO']):
    s = pd.Series(never[ba]['mae'], index=pd.to_datetime(never[ba]['weeks'])) * 100
    ax.plot(s.rolling(8, center=True).mean(), color=plot.SERIES[k], label=ba)
for name, day in eia.EVENTS.items():
    ax.axvline(pd.Timestamp(day), color=plot.MUTED, linewidth=0.8, linestyle='--')
    ax.text(pd.Timestamp(day), ax.get_ylim()[1] * 0.98, ' ' + name, fontsize=8, color=plot.MUTED, va='top')
ax.set_ylabel('weekly error, % (8-week average)'); ax.set_title('Never retrained: error over time')
ax.legend(loc='upper left'); plt.tight_layout()
""")
    md("""
## 1 · Input-drift alarms on weather-driven data

The usual rule is "PSI above 0.2 means significant drift". Here are the actual PSI levels on ordinary weeks:
""")
    code("""
raw = eia.load_raw(); lv = {}
for ba in ('PJM', 'ERCO'):
    s = eia.stream(ba, raw); st = backtest.initial(s)
    weekly = [backtest.score('psi', st, s, b, None, {}, 1) for b in range(60, len(s.X))]
    seasonal = [backtest.score('psi_seasonal', st, s, b, None, {}, 1) for b in range(60, len(s.X))]
    lv[ba] = {'1 week vs training year: median': np.median(weekly), '4 weeks vs same weeks last year: median': np.median(seasonal),
              '4 weeks vs same weeks last year: 10th pct': np.percentile(seasonal, 10)}
pd.DataFrame(lv).round(2)
""")
    md("""
Two reasons the numbers are so far above 0.2:

1. **Too few distinct values.** Several features are daily (yesterday's mean and peak): a week holds only
   7 of them, which cannot fill 10 bins, and every empty bin adds a large fixed amount to PSI.
2. **Weather.** Even comparing 4 weeks with the same 4 weeks a year earlier (5 bins), a hot spell or a
   mild winter shifts the whole demand distribution far more than slow drift does.

So input-drift alarms either fire every week or, set high enough to stay quiet, never fire:
""")
    code("""
ps = t[(t.trigger == 'psi_seasonal') & (t.period == 'test')].groupby('threshold').agg(
    error_pct=('error_mean', lambda x: 100 * x.mean()), retrains_per_operator=('retrains', 'mean'))
ps.round(2)
""")
    md("""
## 2 · Which trigger, chosen on 2020-21, does best on 2022-25?

Price of a retrain = k weeks of the operator's normal forecast error. Savings are against never
retraining, summed over the 12 operators, with a 95% interval from resampling operators.
""")
    code("""
sv = json.load(open('results/real_vs_never.json'))
tab = pd.DataFrame({f'retrain = {k} wk': {trig: f"{v['saving_vs_never']:+.1f}  [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
                                           for trig, v in res.items()} for k, res in sv.items()})
tab.loc[['schedule', 'error', 'page_hinkley', 'error_cp', 'page_hinkley_cp', 'c2st', 'psi_seasonal', 'ks', 'pred_psi', 'psi']]
""")
    code("""
pick = {k: realdata.choose_and_score(t, k).set_index('trigger').threshold for k in realdata.RETRAIN_WEEKS}
pd.DataFrame(pick).rename(columns=lambda k: f'threshold chosen, {k} wk')
""")
    md("""
**Reading it:**

- **Total cost of never retraining is 133.7**, so the best savings are 2-5%. On this task most models
  hold up; retraining pays mainly where demand **grew** (ERCOT, Florida).
- **Performance-based triggers save a little at low and medium retrain prices** and fire rarely
  (ERCOT once or twice). At the highest price the error ratio still saves slightly; Page-Hinkley breaks even.
- **A fixed schedule (every two years) wins when retrains are cheap and loses when they are expensive.**
- **Input-drift triggers:** weekly PSI and prediction PSI fire nearly every week and are very costly;
  seasonal PSI and the KS test, at the thresholds chosen, never fire at all; the classifier test at its
  strictest setting is roughly break-even.
- The chosen thresholds sit at the quiet end of every grid: the tuning period saying "retraining didn't
  pay in 2020-21". That made them slightly too timid for 2022-25, when growth arrived.

## 3 · The change-point trick does not transfer

On the synthetic streams, retraining only on data after the estimated change was the best idea. On
seasonal demand it backfires badly: a few weeks of post-change data lose the yearly cycle, and error
jumps from 4-7% to 6-26%. Page-Hinkley at threshold 1.0, test years:
""")
    code("""
cp = t[(t.period == 'test') & t.trigger.isin(['page_hinkley', 'page_hinkley_cp']) & (t.threshold == 1.0)]
(cp.pivot(index='ba', columns='trigger', values='error_mean') * 100).round(2)
""")
    md("""
## 4 · When did each trigger retrain ERCOT?
""")
    code("""
chosen = realdata.choose_and_score(t, 4).set_index('trigger').threshold
fig, ax = plt.subplots(figsize=(11, 3.2))
names = ['schedule', 'error', 'page_hinkley', 'c2st']
for k, trig in enumerate(names):
    r = next(r for r in rows if r['ba'] == 'ERCO' and r['trigger'] == trig and r['threshold'] == chosen[trig])
    ax.scatter(pd.to_datetime(r['retrains']), [k] * len(r['retrains']), s=40, color=plot.SERIES[k])
ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.set_ylim(-0.5, len(names) - 0.5)
ax.set_xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2026-01-01'))
for name, day in eia.EVENTS.items():
    ax.axvline(pd.Timestamp(day), color=plot.MUTED, linewidth=0.8, linestyle='--')
ax.set_title('ERCOT retrain dates (thresholds chosen on 2020-21, retrain = 4 weeks of error)')
plt.tight_layout()
""")
    md("""
## What this shows

- On real, weather-driven data, **input-drift monitoring as usually configured is noise**: weekly PSI
  sits at 4-7 on ordinary weeks. If you use it, compare like with like (same season), over enough data,
  and expect to set thresholds far above 0.2.
- **Performance triggers are the safe default** when labels come back quickly: they fire rarely and
  save a little at every retrain price.
- **Retraining is worth most where the world grows past the training range** (ERCOT); a tree model can't
  extrapolate, so growth is the drift that matters for this kind of model.
- Ideas that win in simulation (change-point windows) need checking against the real data's structure
  (here, seasonality).
""")
    return c


NOTEBOOKS = {"01": ("01_triggers", nb01), "02": ("02_retrain_actions", nb02), "03": ("03_grid_load", nb03)}

if __name__ == "__main__":
    for key in sys.argv[1:] or NOTEBOOKS:
        name, fn = NOTEBOOKS[key]
        nb = nbf.v4.new_notebook(cells=fn())
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, HERE / f"{name}.ipynb")
        print(HERE / f"{name}.ipynb")
