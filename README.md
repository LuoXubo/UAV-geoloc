# UAV-geoloc

Code for Deep learning based cross-view image matching for UAV geo-localization.

## Getting Started

### Installation

1. Clone the repo

```sh
git clone https://github.com/LuoXubo/UAV-geoloc
```

2. Install requirements

```sh
pip install -r requirements.txt
```

### Data Preparation

1. Download the [UAVDT dataset](https://sites.google.com/site/daviddo0323/projects/uavdt) and extract it to `data/UAV-benchmark-M/`.
2. Arrange the dataset as follows:

```
UAV-geoloc
├── datasets
│   ├── query
│   │   ├── 1
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
│   ├── database
│   │   ├── 1
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
```

### Testing

Test the model with the following commands:

1. Coarse localization

```sh
python test.py --config configs/test_coarse.yaml
```

## Citation

If you find this work useful, please consider citing the following paper:

```
@INPROCEEDINGS{10137193,
  author={Luo, Xubo and Tian, Yaolin and Wan, Xue and Xu, Jingzhong and Ke, Tao},
  booktitle={2022 International Conference on Service Robotics (ICoSR)},
  title={Deep learning based cross-view image matching for UAV geo-localization},
  year={2022},
  volume={},
  number={},
  pages={102-106},
  keywords={Location awareness;Deep learning;Satellites;Service robots;Image matching;Refining;Lighting;deep learning;image matching;geo-localization;autonomous drone navigation},
  doi={10.1109/ICoSR57188.2022.00028}}

```

## Acknowledgements

This repo is built upon the [LPN](https://github.com/wtyhub/LPN) and DFM(https://github.com/ufukefe/DFM). We thank the authors for sharing their code.
