# Real-Time Intermediate Flow Estimation for 3D tomography
## Introduction
This project is a modified implementation of [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/abs/2011.06294), developed within a paper that is currently under review.
More details will be added once the paper will be accepted. 

You may check [this pull request](https://github.com/megvii-research/ECCV2022-RIFE/pull/300) for supporting macOS.
## Usage

### Installation

```
git clone git@github.com:StefanoSanvitoGroup/RIFE-3D-tom
cd RIFE-3D-tom
pip3 install -r requirements.txt
```

* Download the pretrained **HD** models from [here](https://drive.google.com/file/d/1EAbsfY7mjnXNa6RAsATj2ImAEqmHTjbE/view), made available by the RIFE developers

* Unzip and move the pretrained parameters to RIFE-3D-tom/train_log/

* Please note that other pretrained models are available within the [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) and [PracticalRIFE](https://github.com/hzwer/Practical-RIFE) Github pages

### Run

**Image Interpolation**

```
python3 inference_img.py --in_folder '{input_folder}' --add '{num_frames}' --out_folder '{output_folder}' --out_format '{output_format}'
```

Where:
* input_folder is the path to the folder containing the frame sequence that you want to augment
* output_folder is the path where the new sequence of frames will be saved
* num_frames is the number of additional frames that you want to generate between every two frames (please choose 1, 3 or 7)
* output_format is the format (i.e png, tif) in which you want the frames to be saved; if not speicfied, the same format as the input will be used

### Fine Tuning
Copy the pretrained model to RIFE-3D-tom/train_log_original/, the fine tuned model will be saved in RIFE-3D-tom/train_log/


The dataset used for fine tunining should have the following structure:
...


Please note that GPU is required.
```
!torchrun train.py --epoch='{number_of_epochs}' --world_size=1 
```
