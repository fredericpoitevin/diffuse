#!/bin/bash
src_dir="/cstor/stanford/levittm/users/fpoitevi/diffuse/jobs/A3_4/exp/"
trg_dir="./"
#
for file in system.pickle combined_braggmasks.h5
do
  cp ${src_dir}/${file} ${trg_dir}/${file}
done
