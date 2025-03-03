<div align="center">
<h3>Dummy</h3>

Anonymous<sup>1</sup>, Anonymous*<sup>1</sup>

<sup>1</sup>  Anonymous University

DUMMY 2025 Paper: ([arXiv dummy.dummy](about:blank))

<div align="left">

## Abstract


## Preparation


### Environment

    conda create -n DUMMY python=3.10.9 -y
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
    pip install -r requirement.txt

### Dataset preprocess for inference

You can download the datasets from the links below:
([VPO dataset]([https://github.com/cyh-0/CAVP]))
([AVSBench dataset]([https://github.com/OpenNLPLab/AVSBench]))
([IS3 dataset]([[about:blank](https://github.com/kaistmm/SSLalignment)]))


    code will be released soon

The dataset directory should have the following structure:

    ├── datasets
    │   ├── image
    │   │   ├── filename1.jpg
    │   │   ├── filename2.jpg
    │   │   └── ...
    │   ├── wav
    │   │   ├── filename1.wav
    │   │   ├── filename2.wav
    │   │   └── ...
    │   └── gt_seg
    │       ├── filename1.jpg
    │       ├── filename2.jpg
    │       └── ...



## Quick Start


You can easily run the demo by executing the command below:

    python inference_demo.py --methods cross_modal --img_path input.jpg --wav_path input.wav 

## Citation

If this code is useful for your research, please consider citing:

    dummy
