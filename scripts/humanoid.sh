#!/bin/bash

source venv/bin/activate

python -m rlhf.main --env-id Humanoid-v5 --capture-video

deactivate
