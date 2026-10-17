# retrain-trigger

**When should a data scientist be told to retrain a model?**

Models go stale: the world they learned from changes. Monitoring tools usually answer "has the input
data changed (drifted)?". But input data can change without hurting the model, and the model can break
while the inputs look exactly the same. This project measures **retraining triggers by what they cost
and what they save**, against drift whose timing and harm are known, and then on real electricity-demand
forecasting (2019-2025, through COVID and two winter storms).

**Status:** scaffolded 2026-10-17. The synthetic test bench runs; the plan is in [docs/PLAN.md](docs/PLAN.md).

## First result

Total cost (prediction error plus a fixed price per retrain; lower is better), one simulated stream each:

| Trigger | Model's meaning changes | Inputs move somewhere new | Inputs move, harmlessly | Nothing changes |
|---|---|---|---|---|
| never retrain | 105.9 | 71.1 | 26.1 | 25.8 |
| retrain every 10 weeks | 74.9 | 66.4 | 51.8 | 51.7 |
| input drift alarm (PSI) | 105.9 | 51.7 | 36.8 | 25.8 |
| error got worse | 62.6 | 53.9 | **26.1** | **25.8** |

- The input-drift alarm **never fires when the meaning changes** (same inputs, different right answer),
  and **fires for harmless changes**, paying for two retrains that didn't help.
- Watching the model's actual error catches both kinds of real damage and never retrains for nothing
  here. Its catch: the right answers (labels) must come back, and in real life they arrive late.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
pytest -q
python -c "from retrain import simulate, backtest; s = simulate.stream('concept_sudden'); print(backtest.run(s, 'error').retrains)"
```
