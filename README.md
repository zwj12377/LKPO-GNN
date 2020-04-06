# LKPO-GNN
### LKPO-GNN: *Local k-NNs Pattern in Omni-Direction Graph Convolution Neural Network for 3D Point Clouds*


### Introduction
Our method is delicately designed to capture both global and local spatial layout of point cloud by proposing a Local k-NNs Pattern in Omni-Direction Graph Convolution Neural Network architecture, called LKPO-GNN. Our method converts the unordered 3D point cloud into an ordered 1D sequence, to facilitate feeding the raw data into neural networks and simultaneously reducing the computational complexity. LKPO-GNN selects multi-directional k-NNs to form the local topological structure of a centroid, which describes local shapes in the point cloud. Afterwards, GNN is used to combine the local spatial structures and represent the unordered point clouds as a global graph. Experiments on ModelNet40, ShapeNetPart, ScanNet, and S3DIS datasets demonstrate that our proposed method outperforms most existing methods, which verifies the effectiveness and advantage of our work. Additionally, a deep analysis towards illustrating the rationality of our approach, in terms of the learned the topological structure feature, is provided.
### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under Tensorflow1.2 GPU version and Python 2.7 on Ubuntu 16.04. It's highly recommended that you have access to GPUs.

#### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

        TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
        TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
        
Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.

### Dataset

For ModelNet40, ShapeNetPart and ScanNet we use the pre-processed dataset provided by PointNet++ of Charles R. Qi.
Download 3D indoor parsing dataset (S3DIS Dataset) for testing and visualization. Version 1.2 of the dataset is used in this work.


#### Shape Classification

To train a LKPO-GNN model to classify ModelNet40 shapes (using point clouds with XYZ coordinates):

        python train.py

After training, to evaluate the classification accuracies :

        python evaluate.py

#### Object Part Segmentation

To train a model to segment object parts for ShapeNet models:

        cd part_seg
        python train.py


#### Semantic Scene Parsing

        cd LKPO_scannet
        python train.py
