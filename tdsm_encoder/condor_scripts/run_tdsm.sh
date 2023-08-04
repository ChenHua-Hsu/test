#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_102b_cuda/x86_64-centos7-gcc8-opt/setup.sh
python3 -V
python3 /afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/trans_tdsm.py -o '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder' -s 0010