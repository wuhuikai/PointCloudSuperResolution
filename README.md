```
pip install -r requirements.txt

cd code/tf_ops/CD && CUDA_HOME=[CUDA_HOME] bash tf_nndistance_compile.sh
cd code/tf_ops/grouping && CUDA_HOME=[CUDA_HOME] bash tf_grouping_compile.sh
cd code/tf_ops/sampling && CUDA_HOME=[CUDA_HOME] bash tf_sampling_compile.sh

Download training dataset from https://drive.google.com/file/d/1te8d1y2BTFBL_3CB1jpqbOFzkkjvtKsE/view and put it into the folder data/
cd code
python main_gan.py --phase train --dataset ../data/Patches_noHole_and_collected.h5
python main_gan.py --phase train --dataset ../data/Patches_noHole_and_collected.h5 --gan --log_dir ../model/model_res_mesh_pool_gan_ft --batch_size 16 --model_path ../model/model_res_mesh_pool/model-80 --max_epoch 40

python main_gan.py --dataset ../data/test_data/our_collected_data/input --log_dir ../model/model_res_mesh_pool_gan_ft

cd evaluation_code
conda install cgal
mkdir build && cd build && cmake .. && make && mv evaluation ../ && cd .. && rm -rf build

python evaluation_cd.py --pre_path ../model/model_res_mesh_pool_gan_ft/result/input --gt_path ../data/test_data/our_collected_data/gt
python evaluation.py --pre_path ../model/model_res_mesh_pool_gan_ft/result/input --gt_path ../data/test_data/our_collected_data/gt_off --save_path ../model/model_res_mesh_pool_gan_ft/result/input_nuc                                     
```
 
# PCSR
by Huikai Wu 

### Introduction

This repository is for our paper `PCSR`. The code is modified from `PU-Net`. 

### Installation
This repository is based on Tensorflow and the TF operators from `PointNet++`. Therefore, you need to install tensorflow and compile the TF operators. 

The code is tested under TF1.5 (higher version should also work) and Python 3.5 on Ubuntu 16.04.

### Usage

1. Clone the repository:

   ```shell
   git clone ...
   cd PCSR
   ```
2. Compile the TF operators 
   
3. Train the model:
  First, you need to download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/file/d/1te8d1y2BTFBL_3CB1jpqbOFzkkjvtKsE/view?usp=sharing) and put it in folder `data`.
  Then run:
   ```shell
   cd code
   python main_gan.py --phase train ...
   ```

4. Evaluate the model:
   ```shell
   cd code
   python main_gan.py --phase test --log_dir ...
   ```
   You will see the input and output results in the folder `...`.

### Evaluation code
We provide the code to calculate the metric NUC in the evaluation code folder.
   ```shell
   cd evaluation_code
   ...
``` 

## Citation

If PCSR is useful for your research, please consider citing:
