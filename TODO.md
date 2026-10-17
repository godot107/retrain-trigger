# TODO

Built 2026-10-17. Done: synthetic and real studies, weighted change-point window, per-operator thresholds,
weather model with observed temperature and with archived day-ahead forecasts, warm-start docs check,
AWS monitor (Lambda + Scheduler + S3 + SNS, built and parity-checked, not deployed), blog draft, proofread.

## Open
- [ ] Public/private decision, then push to github.com/godot107/retrain-trigger (blog image URLs assume it)
- [ ] AWS: `aws login`, then deploy_cost.sh -> deploy.sh -> `python -m retrain.aws.replay aws ERCO` ->
      screenshot the email page -> down.sh --all, same session
- [ ] Optional: a regional temperature average per multi-state operator (MISO, PJM, SWPP) instead of one city
