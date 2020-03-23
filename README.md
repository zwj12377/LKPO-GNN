# LKPO-GNN
### LKPO-GNN: *Local k-NNs Pattern in Omni-Direction Graph Convolution Neural Network for 3D Point Clouds*


### Introduction
We propose a novel deep LKPO-GNN architecture suitable for extracting the topological features of 3D point cloud by using omni-directional deterministic structured
geometric relationships between the points. Firstly, the centroids were selected from point cloud by FPS. Secondly, the ball query module is used to obtain the rich
local feature of each centroid. Then the LKPO-GNN module extracted the meaningful representation using the spatial layout information of the center points.
After several ball query module and LKPO-GNN modules processing, the center points in 3D point cloud scene becomes less and less, and the feature information of
the center points aggregation become richer. Our method obtains the deeper feature representation of 3D point cloud by connecting the local and global features
of 3D point cloud to improve the classification and segmentation results.
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
