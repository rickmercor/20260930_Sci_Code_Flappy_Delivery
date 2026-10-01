#!/bin/bash
# Harbor verifier entrypoint. grade.py always writes /logs/verifier/reward.json.
mkdir -p /logs/verifier
python3 /tests/grade.py
