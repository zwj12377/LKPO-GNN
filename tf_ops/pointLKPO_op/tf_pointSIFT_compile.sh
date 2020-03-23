#/bin/bash
/usr/local/cuda-8.0/bin/nvcc pointLKPO.cu -o pointLKPO_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
g++ -std=c++11 main.cpp pointLKPO_g.cu.o -o tf_pointLKPO_so.so -shared -fPIC -I /home/zwj/anaconda2/envs/tensorflow/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0