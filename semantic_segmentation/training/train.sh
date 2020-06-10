#!/bin/bash
set -x

LOG="logs/mdl_`date +%Y-%m-%d_%H-%M-%S`.txt"
exec &> >(tee -a "$LOG")

python solve.py
