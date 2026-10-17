# TODO

Built 2026-10-17: synthetic study (6 scenarios x 30 seeds, 13 triggers, 3 retrain actions, label delay),
real EIA-930 study (12 operators, thresholds chosen 2020-21, scored 2022-25), 3 executed notebooks.

## Open
- [ ] Proofread notebooks + README; blog post
- [ ] Public/private decision, then push to github.com/godot107/retrain-trigger
- [ ] Real data: change-point window padded to a full year (keep the seasonal cycle) - may rescue the idea
- [ ] Real data: per-operator thresholds with a longer tuning history; add weather inputs (NOAA) to see
      whether a stronger model drifts differently
- [ ] Warm start: check whether scikit-learn documents/tracks the re-binning behaviour; consider an issue
- [ ] Optional AWS leg: scheduled Lambda performance trigger vs SageMaker Model Monitor (cost-check first)
