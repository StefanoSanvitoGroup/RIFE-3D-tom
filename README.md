# Real-Time Intermediate Flow Estimation for 3D tomography
## Introduction
This project is a modified implementation of [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/abs/2011.06294), developed within a paper that is currently under review.
More details will be added once the paper is be accepted. 

## Usage

### Installation

```
git clone https://github.com/StefanoSanvitoGroup/RIFE-3D-tom.git
cd RIFE-3D-tom
pip3 install -r requirements.txt
```

The installation time on a standard laptop is less than 2 minutes. 

* Download the pretrained **HD** models from [here](https://drive.google.com/file/d/1EAbsfY7mjnXNa6RAsATj2ImAEqmHTjbE/view), made available by the RIFE developers. This is the version used in our paper.

* Unzip and move the pretrained parameters to RIFE-3D-tom/train_log/

* Please note that other pretrained models are available within the [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) and [PracticalRIFE](https://github.com/hzwer/Practical-RIFE) Github repositories.

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


You can test the model on your own data or use the test images in demo/input, kindly provided by Cian Gabbett, Luke Doolan and Jonathan Coleman.
The expected output, i.e. the RIFE-augmented sequence, can be found in demo/expected_output. For this example, we set num_frame=1. The runtime is about 20 seconds for a standard computer.

### Fine Tuning
Copy the pretrained model to RIFE-3D-tom/train_log_original/, the fine tuned model will be saved in RIFE-3D-tom/train_log/


The dataset used for fine tuning should have the same structure as the [Vimeo90K dataset](http://toflow.csail.mit.edu).


Please note that GPU is required for fine tuning.
```
!torchrun train.py --epoch='{number_of_epochs}' --world_size=1 
```

## Citation

If you use this code in your research, please cite our paper:

Gambini, L., Gabbett C., Doolan L., Jones L., Coleman J.N., Gilligan P., & Sanvito, S. Video Frame Interpolation Neural Network for 3D Tomography Across Different Length Scales, Under review in Nature Communications (2024)
