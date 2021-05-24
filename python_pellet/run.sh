#!/bin/bash
if [ "$1" == "1" ]; then
    echo "Install requirements.";
    pip install requirements.txt
    python train.py BattleZoneDeterministic-v0 10000000 4 20000 0
elif [ "$1" == "0" ]; then
    echo "Do not install requirements.";
    python train.py BattleZoneDeterministic-v0 10000000 4 20000 0
else
    echo "Enter either 0 to not install requirements or 1 to install them."
fi