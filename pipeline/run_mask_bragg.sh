#!/bin/bash
#
if [ $# -ne 1 ]; then
  echo "Provide the batch number..."; exit; fi
#
dials.python mask_bragg.py generate /cstor/stanford/levittm/users/fpoitevi/diffuse/jobs/A3_4/exp/system.pickle 3 5 420 $1 
#
