#!/bin/bash
#
nbatch=71 #1 #71
ibatch=71 # 0 #0
while [ $ibatch -le $nbatch ]
do
  if grep -q "elapsed time is" maskbragg${ibatch}.out ; then
    echo "maskbragg${ibatch}.out already done"
  else
    mv maskbragg${ibatch}.out maskbragg${ibatch}.out.bak
    mv maskbragg${ibatch}.err maskbragg${ibatch}.err.bak
    sed "s/XXXibatchXXX/$ibatch/g" launch_mask_bragg.sh > launch_tmp.sh
    sbatch launch_tmp.sh; rm -f launch_tmp.sh
  fi
  ibatch=` expr $ibatch + 1`
done
#
