<div align="center">
<h3>Bridging Audio and Vision: Zero-Shot Audiovisual Segmentation by Connecting Pretrained Models</h3>

Seung-jae Lee, Paul Hongsuck Seo*

  Korea University, South Korea

Interspeech 2025 Paper: [arXiv tba.tba](about:blank)

<div align="left">

## Abstract

Audiovisual segmentation (AVS) aims to identify visual regions corresponding to sound sources, playing a vital role in video understanding, surveillance, and human-computer interaction. Traditional AVS methods depend on large-scale pixel-level annotations, which are costly and time-consuming to obtain. To address this, we propose a novel zero-shot AVS framework that  obviates the need for task-specific supervision by leveraging multiple pretrained models. Our approach integrates audio, vision, and text representations to bridge modality gaps, enabling precise sound source segmentation without AVS-specific annotations. We systematically explore different strategies for connecting pretrained models and evaluate their efficacy across multiple datasets. Experimental results demonstrate that our framework achieves state-of-the-art zero-shot AVS performance, highlighting the effectiveness of multimodal model integration for fine-grained audiovisual segmentation.

## Preparation


### Environment

    conda create -n DUMMY python=3.10.9 -y
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
    pip install -r requirement.txt

### Dataset preprocess for inference

You can download the datasets from the links below:

[VPO dataset](https://github.com/cyh-0/CAVP)

[AVSBench dataset](https://github.com/OpenNLPLab/AVSBench)

[IS3 dataset](https://github.com/kaistmm/SSLalignment)


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

You can evaluate using the command below:

    python eval.py --methods cross_modal --test_data_path /your/dataset/path/


## Citation

If this code is useful for your research, please consider citing:

    @article{lee2025bridging,
    title={Bridging Audio and Vision: Zero-Shot Audiovisual Segmentation by Connecting Pretrained Models},
    author={Lee, Seung-jae and Seo, Paul Hongsuck},
    journal={arXiv preprint arXiv:2506.06537},
    year={2025}
    }
