# High-Fidelity Human Avatars from a Single RGB Camera
### [Project Page](http://cic.tju.edu.cn/faculty/likun/projects/HF-Avatar/)  | [Paper](http://cic.tju.edu.cn/faculty/likun/projects/HF-Avatar/assets/main.pdf) | [Supp](http://cic.tju.edu.cn/faculty/likun/projects/HF-Avatar/assets/supp.pdf)

# Requirement

```
conda create -n Avatar python==3.6.8
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch
or conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

pip install -r requirements.txt

wget https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.4.0.zip
cd pytorch3d
pip install -e .

cd externel

```
Please make sure your gcc version > 7.5 !
# Data Preparation
You first need to run [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) or other pose estimation methods and segmentation or matting method to generate 2d joints and mask to train our model. 
Then the generated data should be organized as follows:
```
--data_dir
----frames_mat
------subject_name
----2d_joints
------subject_name
--------json
----mask_mat
------subject_name
```
We provide the sample data in this link: https://drive.google.com/file/d/1og3eaBTVrvdXaMsMnQv6WHn0II6euxCG/view?usp=sharing
# Training
First, to generate initial geometry by running:
```
python differential_optimization.py --root_dir $data_dir --name $subject_name --device_id $device_id
```
Then, to generate texture map by running:
```
python texture_generation.py --root_dir $data_dir --name $subject_name --device_id $device_id
```

# License
> Copyright 2022 the 3D Vision Group at the College of Intelligence and Computing,  Tianjin University. All Rights Reserved. 
> 
> If you use this code in you work, please cite our publications.
>  
> Permission to use, copy, modify and distribute this software and its documentation for educational, research and non-profit purposes only. 
> Any modification based on this work must be open source and prohibited for commercial use. 
> You must retain, in the source form of any derivative works that you distribute, all copyright, patent, trademark, and attribution notices from the source form of this work. 

# Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{zhao2022avatar,
  author = {Hao Zhao and Jinsong Zhang and Yu-Kun Lai and Zerong Zheng and Yingdi Xie and Yebin Liu and Kun Li},
  title = {High-Fidelity Human Avatars from a Single RGB Camera},
  booktitle = {CVPR},
  year={2022},
}
```

# Acknowlegement
We borrow some code from [NeuralTexture](https://github.com/SSRSGJYD/NeuralTexture). Thanks for their great contribtuions.
