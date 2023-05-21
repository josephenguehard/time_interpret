#!/bin/bash

processes=${processes:-5}
device=${device:-cpu}
seed=${seed:-42}

while [ $# -gt 0 ]
do
  if [[ $1 == *"--"* ]]
  then
    param="${1/--/}"
    declare $param="$2"
  fi
  shift
done

trap ctrl_c INT

function ctrl_c() {
    echo " Stopping running processes..."
    kill -- -$$
}

for lambda_1 in 0.01 0.1 1. 10. 100.
do
  for lambda_2 in 0.01 0.1 1. 10. 100.
  do
    for fold in $(seq 0 4)
    do
      python main.py --explainers extremal_mask --device "$device" --fold "$fold" --seed "$seed" --lambda-1 "$lambda_1" --lambda-2 "$lambda_2" --output-file lambda_study.csv &

      # allow to execute up to $processes jobs in parallel
      if [[ $(jobs -r -p | wc -l) -ge $processes ]]
      then
        # now there are $processes jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
      fi

    done
  done
done

wait