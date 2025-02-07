#!/bin/bash

source venv/bin/activate

python -m rlhf.main --no-synthetic --num-queries 750 --save-model

deactivate
