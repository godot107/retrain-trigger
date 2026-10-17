#!/usr/bin/env bash
# Code bucket, Lambda zip, then the app stack (S3 data bucket, Lambda, weekly schedule, SNS). Safe to re-run.
# Optional: PAGE_EMAIL=you@example.com sends each retrain page by email (confirm the SNS subscription first).
# Pay-per-use only (no always-on resources): see COSTS.md. Tear down with deploy/down.sh.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
require_iam_user

# shellcheck disable=SC2046
aws cloudformation deploy --stack-name "$ARTIFACTS_STACK" --template-file "$ROOT/infra/artifacts.yaml" \
  --no-fail-on-empty-changeset --tags $(tags app)
BUCKET=$(out "$ARTIFACTS_STACK" BucketName)

# Two pure-Python files; a fixed mtime keeps the zip (and its hash) reproducible.
BUILD="$ROOT/build/lambda"
rm -rf "$BUILD" && mkdir -p "$BUILD"
cp "$ROOT/src/retrain/aws/handler.py" "$ROOT/src/retrain/aws/monitor_core.py" "$BUILD/"
(cd "$BUILD" && touch -t 202001010000 ./*.py && zip -qX ../lambda.zip handler.py monitor_core.py)
HASH=$(sha256sum "$ROOT/build/lambda.zip" | cut -c1-12)
CODE_KEY="code/lambda-$HASH.zip"
aws s3 cp "$ROOT/build/lambda.zip" "s3://$BUCKET/$CODE_KEY" --only-show-errors

# shellcheck disable=SC2046
aws cloudformation deploy --stack-name "$APP_STACK" --template-file "$ROOT/infra/app.yaml" \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --tags $(tags app) \
  --parameter-overrides CodeBucket="$BUCKET" CodeKey="$CODE_KEY" Model="${MODEL:-ERCO}" \
    AlertEmail="${PAGE_EMAIL:-}"

echo
echo "Data bucket: $(out "$APP_STACK" BucketName)"
echo "Replay:      PYTHONPATH=src .venv/bin/python -m retrain.aws.replay aws ERCO --weeks 12"
echo "Down:        deploy/down.sh --all"
