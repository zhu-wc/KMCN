# Key features-guided Multi-view Collaborative Network for Image Captioning



This repository contains the reference code for the paper [Key features-guided Multi-view Collaborative Network for Image Captioning](https://www.sciencedirect.com/science/article/pii/S089360802500694X)

![](https://github.com/zhu-wc/KMCN/blob/main/images/overview.jpg)

## Experiment setup

Most of the previous works follow [m2 transformer](https://github.com/aimagelab/meshed-memory-transformer), but they utilized some lower-version packages. Therefore, we recommend  referring to [Xmodal-Ctx](https://github.com/GT-RIPL/Xmodal-Ctx). 

## Data preparation

* **Annotation**. Download the annotation file [annotation.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing). Extarct and put it in the project root directory. Then, some preprocess follow [here](
* **Feature**. We extract grid view with the code in [openal-clip-feature(https://github.com/jianjieluo/OpenAI-CLIP-Feature)](https://github.com/jianjieluo/OpenAI-CLIP-Feature), object view with the code in [vinvl](https://github.com/michelecafagna26/vinvl-visualbackbone), textual view with the code in [haav](https://github.com/GT-RIPL/HAAV/blob/master/ctx/retrieve_captions.py).
* **evaluation**. We use standard evaluation tools to measure the performance of the model, and you can also obtain it [here](https://github.com/luo3300612/image-captioning-DLCT). Extarct and put it in the project root directory.

## Training

```
python train.py --devices 0
```

## Evaluation

```
python test.py
```

## References

[1] [M2](https://github.com/aimagelab/meshed-memory-transformer)

[2] [haav](https://github.com/GT-RIPL/HAAV)

[3] [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)

## Acknowledgements

Thanks the original [m2](https://github.com/aimagelab/meshed-memory-transformer) provided the basic framework of the code. Thanks the project of [haav](https://github.com/GT-RIPL/HAAV) for inspiring our work.





