# [GFRNet](https://arxiv.org/abs/1804.04829)
 Torch implementation for [Learning Warped Guidance for Blind Face Restoration](https://arxiv.org/abs/1804.04829)

# GFRNet framework
<img src="./imgs/architecture/pipeline.jpg">

# Training

Comming Soon.

# Testing

```bash
th test.lua
```
# Models
Download the pre-trained model with the following url and put it into ./checkpoints/FaceRestoration/.
- [BaiduNetDisk](https://pan.baidu.com/s/1q96l3qmTf5Luh-nlqot6Xw)
- [GoogleDrive](https://drive.google.com/open?id=1PhE3Gi9-eHrofyR3LhqEhuVnzh9D7IsX)

# Results
## Restoration on real low quality images
The first row is real low quality image(close-up in right bottom is the guided image). The second row is GFRNet result.

<img src="./imgs/realresults/1.jpg">

## Warped guidance

<img src="./imgs/warpface/warp.jpg">

## IMDB results
 <table style="float:center">
 <tr>
  <th><B>Input</B></th><th><B>Guided Image</B></th><th><B>Bicubic</B></th><th><B>GFRNet Results</B></th>
 </tr>
 <tr>
  <td>
   <img src='./imgs/IMDb/1_2.jpg' >
  </td>
  <td>
  <img src='./imgs/IMDb/1_1.jpg'>
  </td>
  <td>
   <img src='./imgs/IMDb/1_3.jpg'>
  </td>
  <td>
   <img src='./imgs/IMDb/1_4.jpg'>
  </td>
  </tr>
  <tr>
  <td>
   <img src='./imgs/IMDb/2_2.jpg' >
  </td>
  <td>
  <img src='./imgs/IMDb/2_1.jpg'>
  </td>
  <td>
   <img src='./imgs/IMDb/2_3.jpg'>
  </td>
  <td>
   <img src='./imgs/IMDb/2_4.jpg'>
  </td>
  </tr>
 </table>
[More IMDB resutls can be found here.](http://csxmli.xin/GFRNet/)

# Requirements and Dependencies

- [Torch](https://github.com/torch/distro)
- [Cuda](https://developer.nvidia.com/cuda-toolkit-archive)-8.0
- [Stn](https://github.com/qassemoquab/stnbhwd)

# Acknowledgments

Code borrows heavily from [pix2pix](https://github.com/phillipi/pix2pix). Thanks for their excellent work!

