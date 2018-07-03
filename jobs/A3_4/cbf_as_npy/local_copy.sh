#!/bin/bash
src_dir="/cstor/stanford/levittm/users/fpoitevi/diffuse/jobs/A3_4/cbf_as_npy/"
trg_dir="./"
#
for file in A3_4_00450.npy A3_4_00900.npy A3_4_01350.npy A3_4_01800.npy A3_4_02250.npy A3_4_02700.npy A3_4_03150.npy A3_4_03600.npy
do
  cp ${src_dir}/${file} ${trg_dir}/${file}
done
