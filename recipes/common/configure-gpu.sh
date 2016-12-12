#!/bin/bash -e
#
# Tell Theano to use GPU.

THEANO_FLAGS="floatX=float32,device=gpu,nvcc.fastmath=True"
