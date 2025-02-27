Dummy Projectname
=========

This is an official PyTorch implementation of [Dummy Project](about:blank).


Overall Architectures
---------

Preparation
---------

### Environment

    conda create -n DUMMY python=3.10.9 -y
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
    pip install -r requirement.txt

### Dataset preprocess for training

    dummy code

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



Quick Start
---------

You can easily run the demo by executing the command below:

    python inference_demo.py --methods ours --img_path input.jpg --wav_path input.wav 

Citation
---------

    dummy