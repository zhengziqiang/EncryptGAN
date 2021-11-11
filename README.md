# EncryptGAN
## Introduction
EncryptGAN is a [Tensorflow](http://tensorflow.org/)-based framework for training and testing of **[EncryptGAN: Image Steganography with Domain Transform](https://arxiv.org/abs/1905.11582)**

## Citation
If you use this code for your research, please cite our papers.
```
@article{zheng2019encryptgan,
  title={EncryptGAN: Image Steganography with Domain Transform},
  author={Zheng, Ziqiang and Liu, Hongzhi and Yu, Zhibin and Zheng, Haiyong and Wu, Yang and Yang, Yang and Shi, Jianbo},
  journal={arXiv preprint arXiv:1905.11582},
  year={2019}
}
```

## Installation
1. We use [Miniconda3](https://conda.io/miniconda.html) as the basic environment. If you installed the Miniconda3 in path `Conda_Path`, please install `tensorflow-gpu` using the command `Conda_Path/bin/conda install -c anaconda tensorflow-gpu==1.8`.
2. Install dependencies by `Conda_Path/bin/pip install -r requirements.txt` (if necessary). The `requirements.txt` file is provided in this package.

## Train
The training code will be released soon!

## Datasets
- 102Flowers
- CelebA
- Cat2dog

## EncryptGAN settings

<div style="text-align: center" />
<img src="./figures/illustration.jpg" style="max-width: 500px" />
</div>

## Results

### Complex image encryption
<div style="text-align: center" />
<img src="./figures/cat2dog.jpg" style="max-width: 500px" />
</div>


### Exploration of our EncryptGAN
<div style="text-align: center" />
<img src="./figures/visual.jpg" style="max-width: 500px" />
</div>

### More results

<div style="text-align: center" />
<img src="./GIF/cat_64.gif" style="max-width: 500px" />
</div>

<div style="text-align: center" />
<img src="./GIF/dog_128.gif" style="max-width: 500px" />
</div>

### Asymmetric image encryption using private keys
<div style="text-align: center" />
<img src="./GIF/private.gif" style="max-width: 500px" />
</div>
