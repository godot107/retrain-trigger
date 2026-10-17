#!/usr/bin/env bash
# Empty the data bucket and delete the app stack. --all also empties and deletes the code bucket.
# The budget stack is kept.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
require_iam_user
DATA=$(out "$APP_STACK" BucketName 2>/dev/null || true)
if [ -n "$DATA" ] && [ "$DATA" != "None" ]; then
  aws s3 rm "s3://$DATA" --recursive --only-show-errors
fi
aws cloudformation delete-stack --stack-name "$APP_STACK"
aws cloudformation wait stack-delete-complete --stack-name "$APP_STACK"
echo "Deleted $APP_STACK."
if [ "${1:-}" = "--all" ]; then
  BUCKET=$(out "$ARTIFACTS_STACK" BucketName)
  aws s3 rm "s3://$BUCKET" --recursive --only-show-errors
  aws cloudformation delete-stack --stack-name "$ARTIFACTS_STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$ARTIFACTS_STACK"
  echo "Deleted $ARTIFACTS_STACK and its bucket."
fi
