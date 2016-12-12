#!/bin/bash -e
#
# Tell Theano to use the new GpuArray backend.

THEANO_FLAGS="floatX=float32,device=cuda0,contexts=dev0->cuda0;dev1->cuda1,nvcc.fastmath=True"
