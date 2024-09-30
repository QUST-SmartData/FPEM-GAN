# FPEM-GAN

This is the official repository for "Efficient Image Inpainting of Microresistivity Logs: A DDPM-Based Pseudo-Labeling Approach with FPEM-GAN". Please cite this work if you find this repository useful for your project.


In the process of geophysical exploration, the logging image is incomplete due to the mismatch between the size of the logging instrument and the size of the borehole. The missing data will seriously affect the geologic analysis. However, due to the situation of lacking complete images as training labels, existing methods are generally based on usually used algorithms or unsupervised learning methods, which bring abundant computation and time-consuming. In addition, the results match not very well as high-angle fractures appeared a lot and also the fine-grained texture in the inpainted regions. It significantly affects the discrimination of the interpretation for geological phenomena. To solve the aboved problem, we propose a deep learning method to inpaint strati-graphic features. First, to conduct the consuming time and according to the issues of less labels for training, we proposed a new method with pseudo-labeled training datasets in the inpainting process. Second, in order to improve the accuracy of inpainting to high-angle fractures, we also proposed a Fusion-Perspective-Enhancement Module (FPEM), which can effectively infer the missing regions based on the contextual guidance. Finally, to better describe the fine-grained texture, we proposed a new discriminator called SM-Unet, which help enhancing much more textured features highlighting to the fine-grained, the new discriminator can adjust the weight of various regions through producing soft labeling during the training procedure. The Peak Signal-to-Noise Ratio of the proposed algorithm in the logging image dataset is 25.35, the highest Structural Similarity Index is 0.901. Compared to the state-of-the-art methods, the proposed method shows good results matching very well especially for high-angle fractures and fine-grained textured features, and costs less computation.


![Image text](https://github.com/ZZY19980203/FPEM-GAN/blob/main/img_folder/framework.jpg)


## Prerequisites

- Python 3.8
- PyTorch 1.12.0 + cu113 
- NVIDIA GPU + CUDA cuDNN

## Installation

- Clone this repo:

  ```
  git clone https://github.com/ZZY19980203/FPEM-GAN.git
  ```

- Install PyTorch and dependencies from [http://pytorch.org](http://pytorch.org/)

- Install python requirements:

  ```
  pip install -r requirements.txt
  ```

## Started



###  generate reliable pseudo-labels

1、First, we need to generate reliable pseudo-labels by teacher model, the generation model is based on DDPM, and we need to download the corresponding pre-trained model.

```
cd teacher_model
pip install --upgrade gdown && bash ./download.sh
```

2、We need to crop the original logging image. The logging image is first cropped into multiple images of equal length and width, and then resized to 256 x 256.

3、running code.  In the .yam file inside the confs, you need to set the location of the original logging image and mask, and you also need to set the location where the image will be saved.

```
python test.py --conf_path confs/test_inet256_thin.yml
```



### Training and testing student models

#### Trainting

1、Training the student model also requires a configuration .yml file, which we save in the checkpoint file.

2、Our FPEM-GAN is trained in the same way as EdgeConnect in three stages: 1) training the edge model; 2) training the inpainting model; 3) training the joint model.

```
python train.py --model [stage] --checkpoints [path to checkpoints]
```

#### Testing

You can test the model on all three stages: 1) edge model, 2) inpaint model and 3) joint model. In each case, you need to provide an input image (image with a mask) and a grayscale mask file. Please make sure that the mask file covers the entire mask region in the input image. To test the model:

```
python test.py --model [stage] --checkpoints [path to checkpoints] --input [path to input directory or file] --mask [path to masks directory or mask file] --output [path to the output directory]
```

## Example
Below is a display of some of our results, the original images and results can be viewed in the file ``` /img_folder/example/ ```.

![Image text](https://github.com/ZZY19980203/FPEM-GAN/blob/main/img_folder/exmple.jpg)

## Acknowledgements

We would like to thank [edge-connect](https://github.com/knazeri/edge-connect), [RePaint](https://github.com/andreas128/RePaint) and [guided-diffuion](https://github.com/openai/guided-diffusion.git).

If we missed a contribution, please contact us.
