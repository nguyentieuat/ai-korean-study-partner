#!/usr/bin/env bash
set -e
export $(grep -v '^#' .env | xargs)
uvicorn app:app --host 0.0.0.0 --port 5006 --workers 1
