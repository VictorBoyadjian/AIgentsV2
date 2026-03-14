#!/bin/bash
# =============================================================================
# Start Celery worker + beat in a single process
# Used for Railway deployment where we run worker as a separate service
#
# On Railway:
#   - Create a second service pointing to the same repo
#   - Set its start command to: bash start_worker.sh
#   - Ensure REDIS_URL and DATABASE_URL are shared variables
# =============================================================================

set -e

echo "Starting Celery worker + beat..."
echo "  REDIS_URL: ${REDIS_URL:-(not set, using defaults)}"
echo "  DATABASE_URL: ${DATABASE_URL:+(set)}"

exec celery -A orchestration.task_queue.celery_app worker \
    --beat \
    --loglevel=info \
    --concurrency=2
