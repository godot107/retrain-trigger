# TODO

- [x] Scaffold: simulator (5 scenarios), detectors, walk-forward backtest, day-one synthetic table
- [ ] Many seeds + intervals; cost curves over retrain cost; label-delay sensitivity
- [ ] More detectors: MMD / classifier two-sample, ADWIN / Page-Hinkley on errors, "input flags + performance confirms"
- [ ] Retrain actions: refit vs hyperparameter re-tune vs warm start, per scenario
- [ ] EIA-930 day-ahead load forecaster (per BA), weekly walk-forward 2019-2025, real events as checkpoints
- [ ] Notebooks + README results + blog; "which trigger when" decision chart
- [ ] Optional AWS leg: scheduled Lambda trigger vs SageMaker Model Monitor (cost-check first, tear down same session)
