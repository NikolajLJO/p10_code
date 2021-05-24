#!/bin/bash
if [ "$1" == "1" ]; then
    echo "Install requirements.";
    pip3 install -r requirements.txt
    python3 train.py BattleZoneDeterministic-v0 10000000 4 20000 0
elif [ "$1" == "0" ]; then
    echo "Do not install requirements.";
    python3 train.py BattleZoneDeterministic-v0 10000000 4 20000 0
else
    echo "Enter either 0 to not install requirements or 1 to install them."
fi