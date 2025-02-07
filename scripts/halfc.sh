#!/bin/bash

source venv/bin/activate

python -m rlhf.main --capture-video --save-model

deactivate
