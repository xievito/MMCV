# Multi-Mode Cross-View Mutual Learning for Enhanced Self-Supervised 3D Action Recognition
Qianbin Xie, Xun Gong, Xinyun Zou

This repository includes Python (PyTorch) implementation of the MMCV.

![](./images/mmcv.png)

## Abstract
Self-supervised learning has emerged as a promising approach for 3D action recognition, addressing challenges associated with unlabeled data. 
However, existing methods often struggle with integrating complementary information across modalities and fine-grained optimization of 
intra-modal features. This paper introduces a novel framework, Multi-Mode Cross-View Mutual Learning (MMCV), which combines cross-view 
contrastive learning (CVCL) and cross-view knowledge distillation (CVKD). CVCL diversifies negative samples using two distinct momentum 
encoders, enhancing the InfoNCE loss from both inter-modal and inter-model perspectives. CVKD ensures semantic consistency by facilitating 
knowledge transfer across models and modalities. Our framework achieves state-of-the-art performance on NTU RGB+D 60, NTU RGB+D 120, and 
PKU-MMD II datasets, demonstrating superior generalization capabilities. Here we show that MMCV significantly improves the discriminability 
of inter-modality representations, achieving an accuracy of 91.0%, 77.5%, and 54.5% on NTU-60 (x-view), NTU-120 (x-set), and PKU-II datasets,
respectively. Our findings highlight the potential of MMCV in advancing self-supervised 3D action recognition.
## Requirements

```bash
python==3.8.13
torch==1.8.1+cu111
torchvision==0.9.1+cu111
tensorboard==2.9.0
pandas==1.4.3
scikit-learn==1.1.1
tqdm==4.64.0
numpy==1.22.4
```

## Data Preprocessing

Download the raw data of [NTU_RGB+D](https://github.com/shahroudy/NTURGB-D) and [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html) and save to ./data folder
>cd data_gen \
>python ntu_gendata.py
## Training and Testing
Please refer to the bash scripts

## Pretrained Models
NTU-60 and NTU-120: [pretrained_models](https://rec.ustc.edu.cn/share/5f6a5ee0-01dd-11ed-b9ae-8301ca6d3d37)

## Citation
If you find this work useful for your research, please consider citing our work:
```
@article{Xie_2025_MMCV,
    title={Multi-Mode Cross-View Mutual Learning for Enhanced Self-Supervised 3D Action Recognition},
    author={Qianbin Xie, Xun Gong, Xinyun Zou},
    journal={The Visual Computer},
    year={2025}
}
```

## Acknowledgment
The framework of our code is based on [CMD](https://github.com/maoyunyao/CMD).

