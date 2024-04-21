# YOLOv9

This is a pip package compatible adaptation of the official [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://github.com/WongKinYiu/yolov9) reposistory.


[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2402.13616-B31B1B.svg)](https://arxiv.org/abs/2402.13616)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kadirnar/Yolov9)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/merve/yolov9)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov9-object-detection-on-custom-dataset.ipynb)
[![OpenCV](https://img.shields.io/badge/OpenCV-BlogPost-black?logo=opencv&labelColor=blue&color=black)](https://learnopencv.com/yolov9-advancing-the-yolo-legacy/)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>


## Performance 

MS COCO

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | Param. | FLOPs |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv9-T**]() | 640 | **38.3%** | **53.1%** | **41.3%** | **2.0M** | **7.7G** |
| [**YOLOv9-S**]() | 640 | **46.8%** | **63.4%** | **50.7%** | **7.1M** | **26.4G** |
| [**YOLOv9-M**]() | 640 | **51.4%** | **68.1%** | **56.1%** | **20.0M** | **76.3G** |
| [**YOLOv9-C**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt) | 640 | **53.0%** | **70.2%** | **57.8%** | **25.3M** | **102.1G** |
| [**YOLOv9-E**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt) | 640 | **55.6%** | **72.8%** | **60.6%** | **57.3M** | **189.0G** |
<!-- | [**YOLOv9 (ReLU)**]() | 640 | **51.9%** | **69.1%** | **56.5%** | **25.3M** | **102.1G** | -->

<!-- tiny, small, and medium models will be released after the paper be accepted and published. -->

## Useful Links

<details><summary> <b>Expand</b> </summary>

Custom training: https://github.com/WongKinYiu/yolov9/issues/30#issuecomment-1960955297
    
ONNX export: https://github.com/WongKinYiu/yolov9/issues/2#issuecomment-1960519506 https://github.com/WongKinYiu/yolov9/issues/40#issue-2150697688 https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461

ONNX export for segmentation: https://github.com/WongKinYiu/yolov9/issues/260#issue-2191162150

TensorRT inference: https://github.com/WongKinYiu/yolov9/issues/143#issuecomment-1975049660 https://github.com/WongKinYiu/yolov9/issues/34#issue-2150393690 https://github.com/WongKinYiu/yolov9/issues/79#issue-2153547004 https://github.com/WongKinYiu/yolov9/issues/143#issue-2164002309

QAT TensorRT: https://github.com/WongKinYiu/yolov9/issues/327#issue-2229284136 https://github.com/WongKinYiu/yolov9/issues/253#issue-2189520073

OpenVINO: https://github.com/WongKinYiu/yolov9/issues/164#issue-2168540003

C# ONNX inference: https://github.com/WongKinYiu/yolov9/issues/95#issue-2155974619

C# OpenVINO inference: https://github.com/WongKinYiu/yolov9/issues/95#issuecomment-1968131244

OpenCV: https://github.com/WongKinYiu/yolov9/issues/113#issuecomment-1971327672

Hugging Face demo: https://github.com/WongKinYiu/yolov9/issues/45#issuecomment-1961496943

CoLab demo: https://github.com/WongKinYiu/yolov9/pull/18

ONNXSlim export: https://github.com/WongKinYiu/yolov9/pull/37

YOLOv9 ROS: https://github.com/WongKinYiu/yolov9/issues/144#issue-2164210644

YOLOv9 ROS TensorRT: https://github.com/WongKinYiu/yolov9/issues/145#issue-2164218595

YOLOv9 Julia: https://github.com/WongKinYiu/yolov9/issues/141#issuecomment-1973710107

YOLOv9 MLX: https://github.com/WongKinYiu/yolov9/issues/258#issue-2190586540

YOLOv9 StrongSORT with OSNet: https://github.com/WongKinYiu/yolov9/issues/299#issue-2212093340

YOLOv9 ByteTrack: https://github.com/WongKinYiu/yolov9/issues/78#issue-2153512879

YOLOv9 DeepSORT: https://github.com/WongKinYiu/yolov9/issues/98#issue-2156172319

YOLOv9 counting: https://github.com/WongKinYiu/yolov9/issues/84#issue-2153904804

YOLOv9 face detection: https://github.com/WongKinYiu/yolov9/issues/121#issue-2160218766

YOLOv9 segmentation onnxruntime: https://github.com/WongKinYiu/yolov9/issues/151#issue-2165667350

Comet logging: https://github.com/WongKinYiu/yolov9/pull/110

MLflow logging: https://github.com/WongKinYiu/yolov9/pull/87

AnyLabeling tool: https://github.com/WongKinYiu/yolov9/issues/48#issue-2152139662

AX650N deploy: https://github.com/WongKinYiu/yolov9/issues/96#issue-2156115760

Conda environment: https://github.com/WongKinYiu/yolov9/pull/93

AutoDL docker environment: https://github.com/WongKinYiu/yolov9/issues/112#issue-2158203480

</details>

## Installing
To install yolov9 as a package from PyPI:

``` shell
pip install yolov9py
```

## Model API

The package comes with an additional load utility:

``` shell
from yolov9.load import load_model

model_cfg_path = ...
ckpt_path = ...
n_classes = 3
n_channels = 3

# This returns a nn.Module torch model
model = load_model(model_cfg_path, weights=ckpt_path, classes=n_classes, channels=n_channels)

```

## Citation

```
@article{wang2024yolov9,
  title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
  booktitle={arXiv preprint arXiv:2402.13616},
  year={2024}
}
```

```
@article{chang2023yolor,
  title={{YOLOR}-Based Multi-Task Learning},
  author={Chang, Hung-Shuo and Wang, Chien-Yao and Wang, Richard Robert and Chou, Gene and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2309.16921},
  year={2023}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)

</details>
