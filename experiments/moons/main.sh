#!/bin/bash

device=${device:-cpu}

while [ $# -gt 0 ]
do
  if [[ $1 == *"--"* ]]
  then
    param="${1/--/}"
    declare $param="$2"
  fi
  shift
done

for seed in $(seq 12 12 120)
do
  python main.py --device "$device" --seed "$seed"
done

for seed in $(seq 12 12 120)
do
  python main.py --device "$device" --seed "$seed" --softplus
done
