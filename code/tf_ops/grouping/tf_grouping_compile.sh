#!/usr/bin/env bash
${CUDA_HOME}/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I ${CUDA_HOME}/include -lcudart -L ${CUDA_HOME}/lib64/ -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I${TF_INC}/external/nsync/public -L${TF_LIB} -I${TF_INC}
