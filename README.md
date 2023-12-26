## Installation

Install Pytorch and Tensorflow (for TensorBoard). You'll need to have access to GPUs. The code is tested with Ubuntu 18.04, Pytorch v1.16, TensorFlow v1.14, CUDA 10.1 and cuDNN v7.4.

Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for using GraspNet-1Billion dataset.
```bash
cd graspnetAPI
pip install .
```

## Training

CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense --log_dir logs/log_rs --batch_size 2 --dataset_root /data/Benchmark/graspnet

## Testing

CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs --checkpoint_path logs/log_rs/checkpoint.tar --camera realsense --dataset_root /data/Benchmark/graspnet
