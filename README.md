# Multi-Fiber Networks for Video Recognition
This repository contains the code and trained models of:

Yunpeng Chen, Yannis Kalantidis, Jianshu Li, Shuicheng Yan, Jiashi Feng. "Multi-Fiber Networks for Video Recognition" ([PDF](http://arxiv.org/abs/1807.11195)).


## Implementation

We use [MXNet \@92053bd](https://github.com/cypw/mxnet/tree/92053bd3e71f687b5315b8412a6ac65eb0cc32d5) for image classification and [PyTorch 0.4.0a0\@a83c240](https://github.com/pytorch/pytorch) for video classification.

### Normalization
The inputs are substrated by mean RGB = [ 124, 117, 104 ], and then multiplied by 0.0167.


## Usage

Train motion from scratch:
```
python train_kinetics.py
```

Fine-tune with pre-trained model:
```
python train_ucf101.py
```
or 
```
python train_hmdb51.py
```

Evaluate the trained model:
```
cd test
# the default setting is to test trained model on ucf-101 (split1)
python evaluate_video.py
```


## Results

### Image Recognition (ImageNet-1k)

**Single Model, Single Crop Validation Accuracy:**

Model                   |  Params  |  FLOPs  |  Top-1  |  Top-5  |         MXNet Model
:-----------------------|:--------:|:-------:|:-------:|:-------:|------------------------------------:
ResNet-18 (reproduced)  |  11.7 M  |  1.8 G  |  71.4 % |  90.2 % | [GoogleDrive](https://goo.gl/QSkx8S)
ResNet-18 (MF embedded) |   9.6 M  |  1.6 G  |  74.3 % |  92.1 % | [GoogleDrive](https://goo.gl/Myq5Wh)
MF-Net (N=16)           |   5.8 M  |  861 M  |  74.6 % |  92.0 % | [GoogleDrive](https://goo.gl/53Gfsg)


### Video Recognition (UCF-101, HMDB51, Kinetics)

Model         | Params |  Target Dataset   |  Top-1
:-------------|:------:|:-----------------:|:-------:
MF-Net \(3D\) |  8.0 M |     Kinetics      | 72.8 %
MF-Net \(3D\) |  8.0 M |     UCF-101       | 96.0 %*
MF-Net \(3D\) |  8.0 M |      HMDB51       | 74.6 %*

\* accuracy averaged on slip1, slip2, and slip3.


## Trained Models

Model         |  Target Dataset  |            PyTorch Model
:-------------|:----------------:|:----------------------------------:
MF-Net \(2D\) |   ImageNet-1k    |[GoogleDrive](https://goo.gl/h5jG3B)
MF-Net \(3D\) |     Kinetics     |[GoogleDrive](https://goo.gl/QdE85T)
MF-Net \(3D\) | UCF-101 (split1) |[GoogleDrive](https://goo.gl/mML2gv)
MF-Net \(3D\) |  HMDB51 (split1) |[GoogleDrive](https://goo.gl/cD4hnw)


## Other Resources

ImageNet-1k Trainig/Validation List:
- Download link: [GoogleDrive](https://goo.gl/Ne42bM)

ImageNet-1k category name mapping table:
- Download link: [GoogleDrive](https://goo.gl/YTAED5)

Kinetics Dataset:
- Downloader: [GitHub](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)

UCF-101 Dataset:
- Download link: [Website](http://crcv.ucf.edu/data/UCF101.php)

HMDB51 Dataset:
- Download link: [Website](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database)


## FAQ

### Do I need to convert the raw videos to specific format?
- Our `dataiter' supports reading from raw videos and can tolerate corrupted videos.

### How can I make the training faster?
- Decoding frames from compressed videos consumes quite a lot CPU resources which is the bottleneck for the speed. You can try to convert the downloaded videos to other format or reduce the quality of the video. For example:
```
# convet to sort_edge_length = 360
ffmpeg -y -i ${SRC_VID} -c:v mpeg4 -filter:v "scale=min(iw\,(360*iw)/min(iw\,ih)):-1" -b:v 640k -an ${DST_VID}
# or, convet to sort_edge_length = 256
ffmpeg -y -i ${SRC_VID} -c:v mpeg4 -filter:v "scale=min(iw\,(256*iw)/min(iw\,ih)):-1" -b:v 512k -an ${DST_VID}
# or, convet to sort_edge_length = 160
ffmpeg -y -i ${SRC_VID} -c:v mpeg4 -filter:v "scale=min(iw\,(160*iw)/min(iw\,ih)):-1" -b:v 240k -an ${DST_VID}
```
- Find another computer with better CPU.
- The group convolution may not be well optimized.



## Citation
If you use our code/model in your work or find it is helpful, please cite the paper:
```
@inproceedings{chen2018multifiber,
  title={Multi-Fiber networks for Video Recognition},
  author={Chen, Yunpeng and Kalantidis, Yannis and Li, Jianshu and Yan, Shuicheng and Feng, Jiashi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2018}
}
```
