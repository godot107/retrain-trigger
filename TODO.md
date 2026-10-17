# TODO

Built 2026-10-17. Done from the previous list: padded (weighted) change-point window; warm-start check
(documented in the `warm_start` docstring, no run-time warning: no issue to file); per-operator thresholds
(worse than shared); weather inputs (stronger model drifts faster; classifier input-drift trigger wins);
README + notebooks updated; blog.md drafted.

## Open
- [ ] Proofread blog.md and notebooks
- [ ] Public/private decision, then push to github.com/godot107/retrain-trigger (blog image URLs assume it)
- [ ] Weather: use archived *forecasts* instead of observed temperature (Open-Meteo historical forecast
      API) to remove the perfect-forecast assumption
- [ ] Optional AWS leg: scheduled Lambda performance/classifier trigger vs SageMaker Model Monitor
      (cost-check first, tear down same session; needs `aws login`)
