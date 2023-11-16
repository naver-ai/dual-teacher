<div align="center">

## Switching Temporary Teachers for Semi-Supervised Semantic Segmentation
  
[![PWC](https://img.shields.io/badge/NeurIPS%20-2023-8A2BE2)](https://nips.cc/virtual/2023/poster/72052)
</div>

> **Switching Temporary Teachers for Semi-Supervised Semantic Segmentation**<br>
> [Jaemin Na](https://najaemin92.github.io), [Jung-Woo Ha](https://scholar.google.com/citations?user=eGj3ay4AAAAJ&hl), [Hyung Jin Chang](https://hyungjinchang.wordpress.com), [Dongyoon Han*](https://dongyoonhan.github.io/), and [Wonjun Hwang*](https://scholar.google.co.uk/citations?user=-I8AfBAAAAAJ&hl=en).<br>
> In NeurIPS 2023.<br><br/>

<div align=center><img src="https://github.com/NaJaeMin92/Dual-Teacher/blob/main/main_fig.png" width="60%"></div><br/>

<!-- [YouTube](https://www.youtube.com/watchwatch?v=o0jEox4z3OI)<br> -->
> **Abstract:** *The teacher-student framework, prevalent in semi-supervised semantic segmentation, mainly employs the exponential moving average (EMA) to update a single teacher's weights based on those of the student. However, EMA updates raise a problem in that the weights of the teacher and student are getting coupled, causing a potential performance bottleneck. Furthermore, this problem may get severer when training with more complicated labels such as segmentation masks but with few annotated data. This paper introduces Dual Teacher, a simple yet effective approach that employs dual temporary teachers aiming to the student to alleviate the coupling problem. The temporary teachers work in shifts and are progressively improved, so consistently keep the teacher and student from becoming excessively close. Specifically, the temporary teachers periodically take turns generating pseudo-labels to train a student model and keep the distinct characteristics of the student model for each epoch. Consequently, Dual Teacher achieves competitive performance on the PASCAL VOC, Cityscapes, and ADE20K benchmarks with remarkably shorter training times than state-of-the-art methods. Moreover, we demonstrate that our approach is model-agnostic and compatible with both CNN- and Transformer-based models.*


## Dataset
Download [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K) dataset and modify your path in [configuration file](https://github.com/NaJaeMin92/Dual-Teacher/blob/d3f177e727d93879ab09d9ad5e99a33151142d28/local_configs/_base_/datasets/ade20k_repeat.py#L3).  
For semi-supervised learning scenarios, split the images based on the partitions of the text files in the [ADEChallengeData2016](https://github.com/NaJaeMin92/Dual-Teacher/tree/d3f177e727d93879ab09d9ad5e99a33151142d28/data/ADEChallengeData2016).
```
├── ./data
    ├── ADEChallengeData2016
        ├── images
          ├── training631_l
          ├── training631_u
        ├── annotations
          ├── training631_l
          ├── training631_u
```

## Installation
For installation, please refer to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Other requirements:
```pip install timm==0.3.2```

An example (works for me): ```CUDA 10.1``` and  ```pytorch 1.7.1``` 

```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
cd Dual-Teacher && pip install -e . --user
```

## Training

Download `initial weights` 
(
[google drive](https://drive.google.com/file/d/1TKC3ajdhRmSgrqW6Bnl0PpS9mN3UpM4n/view?usp=share_link)
) 
pretrained on ImageNet-1K, and put them in a folder ```pretrained/```.  

Modify `img_dir` and `ann_dir` according to the partitions in [configuration file](https://github.com/NaJaeMin92/Dual-Teacher/blob/d3f177e727d93879ab09d9ad5e99a33151142d28/local_configs/_base_/datasets/ade20k_repeat.py#L50).
```
bash dist_train.sh # Multi-gpu training
```
## License

Please find the [LICENSE](https://github.com/NaJaeMin92/Dual-Teacher/blob/main/LICENSE) file. This code, built on the [SegFormer codebase](https://github.com/NVlabs/SegFormer), adheres to the same license.

## Citation
```bibtex  
@inproceedings{na2023switching,
  title={Switching Temporary Teachers for Semi-Supervised Semantic Segmentation},
  author={Jaemin Na and Jungwoo Ha and Hyungjin Chang and Dongyoon Han and Wonjun Hwang},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## Contact
For questions, please contact: osial46@ajou.ac.kr

