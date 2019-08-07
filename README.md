# Point Cloud Super Resolution with Adversarial Residual Graph Networks
[[arXiv]](https://arxiv.org/abs/1908.02111) [[Home]](http://wuhuikai.me/)

Official implementation of **Point Cloud Super Resolution with Adversarial Residual Graph Networks**. 
```
@inproceedings{wu2019point,
  title     = {Point Cloud Super Resolution with Adversarial Residual Graph Networks},
  author    = {Wu, Huikai and Zhang, Junge and Huang, Kaiqi},
  booktitle = {arXiv preprint arXiv:1908.02111},
  year      = {2019}
}
```
Contact: Hui-Kai Wu (huikaiwu@icloud.com)
## Install
The code is tested with `TF1.5` (higher version should also work) and `Python 3.5` on `Ubuntu 16.04`

1. Clone the repository:

   ```shell
   git clone https://github.com/wuhuikai/PointCloudSuperResolution
   cd PointCloudSuperResolution
   ```

2. Install Requirements

    ```shell
    pip install -r requirements.txt
    ```

3. Compile the TF operators

    ```shell
    cd code/tf_ops/CD && CUDA_HOME=[CUDA_HOME] bash tf_nndistance_compile.sh
    cd code/tf_ops/grouping && CUDA_HOME=[CUDA_HOME] bash tf_grouping_compile.sh
    cd code/tf_ops/sampling && CUDA_HOME=[CUDA_HOME] bash tf_sampling_compile.sh
    ``` 

## Train & Test   
1. Download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/file/d/1te8d1y2BTFBL_3CB1jpqbOFzkkjvtKsE/view?usp=sharing) and put it in folder `data`.

2. Train [Optional]

   ```shell
   cd code
   python main_gan.py --phase train --dataset ../data/Patches_noHole_and_collected.h5
   python main_gan.py --phase train --dataset ../data/Patches_noHole_and_collected.h5 --gan --log_dir ../model/model_res_mesh_pool_gan_ft --batch_size 16 --model_path ../model/model_res_mesh_pool/model-80 --max_epoch 40
   ```

3. Predict
   ```shell
   python main_gan.py --dataset ../data/test_data/our_collected_data/input --log_dir ../model/model_res_mesh_pool_gan_ft
   ```

4. Evluation
   
   ```shell
   cd evaluation_code
   conda install cgal
   mkdir build && cd build && cmake .. && make && mv evaluation ../ && cd .. && rm -rf build

   python evaluation_cd.py --pre_path ../model/model_res_mesh_pool_gan_ft/result/input --gt_path ../data/test_data/our_collected_data/gt
   python evaluation.py --pre_path ../model/model_res_mesh_pool_gan_ft/result/input --gt_path ../data/test_data/our_collected_data/gt_off --save_path ../model/model_res_mesh_pool_gan_ft/result/input_nuc
   ```

## Acknowledgement
The code is modified from [PointNet++](https://github.com/charlesq34/pointnet2) and [PU-Net](https://github.com/yulequan/PU-Net).
