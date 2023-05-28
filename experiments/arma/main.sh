#!/bin/bash

processes=${processes:-10}
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

for rare_dim in $(seq 1 2)
do
  for fold in $(seq 0 4)
  do
    python main.py --rare-dim "$rare_dim" --device "$device" --fold "$fold" --seed "$seed" &

    # Support lower versions
    if ((BASH_VERSINFO[0] >= 4)) && ((BASH_VERSINFO[1] >= 3))
    then
      # allow to execute up to $processes jobs in parallel
      if [[ $(jobs -r -p | wc -l) -ge $processes ]]
      then
        # now there are $processes jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
      fi
    else
      # allow to execute up to $processes jobs in parallel
      while [[ $(jobs -r -p | wc -l) -ge $processes ]]
      do
        # now there are $processes jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        sleep 1
      done
    fi

  done
done

wait