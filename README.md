# Shrinking unit: a graph convolution-based unit for CNN-like 3D point cloud feature extractors
Created by <a href="https://albertotamajo.github.io/" target="_blank">Alberto Tamajo</a>, <a href="https://i3mainz.hs-mainz.de/team/bastianplass/" target="_blank">Bastian Plaß</a> and <a href="https://i3mainz.hs-mainz.de/team/thomasklauer/" target="_blank">Thomas Klauer</a>.

<p align="center">
  <img src="https://github.com/albertotamajo/Shrinking-unit/blob/main/doc/ShrinkingUnitIllustrated.png", width="40%", height="40%"/>
  <img src="https://github.com/albertotamajo/Shrinking-unit/blob/main/doc/ShrinkingNet.png", width="50%", height="50%"/>
</p>

### Introduction
<p align="justify">
This work is based on our <a href="https://arxiv.org/abs/2209.12770">arXiv manuscript</a>, which is going to be submitted to <a href="https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34">IEEE Transactions on Pattern Analysis and Machine Intelligence</a>. We propose a graph convolution-based unit, dubbed Shrinking unit, that can be stacked vertically and horizontally for the design of CNN-like 3D point cloud feature extractors.
</p>
<p align="justify">
3D point clouds have attracted increasing attention in architecture, engineering, and construction due to their high- quality object representation and efficient acquisition methods. Consequently, many point cloud feature detection methods have been proposed in the literature to automate some workflows, such as their classification or part segmentation. Nevertheless, the performance of point cloud automated systems significantly lags below their image counterparts. While part of this failure stems from the irregularity, unstructuredness and disorder of point clouds, which makes the task of point cloud feature detection significantly more challenging than the image one, we argue that a lack of inspiration from the image domain might be the primary cause of such a gap. Indeed, given the overwhelming
success of Convolutional Neural Network (CNN)s in image feature detection, it seems reasonable to design their point cloud counterparts, but none of the proposed approaches closely resembles them. Specifically, even though many approaches generalise the convolution operation in point clouds, they fail to emulate the
CNNs multiple-feature detection and pooling operations. For this reason, we propose a graph convolution-based unit, dubbed Shrinking unit, that can be stacked vertically and horizontally for the design of CNN-like 3D point cloud feature extractors. Given that self, local and global correlations between points in a point cloud convey crucial spatial geometric information, we also leverage them during the feature extraction process. We evaluate our proposal by designing a feature extractor model for the ModelNet-10 benchmark dataset and achieve 90.64% classification accuracy, demonstrating that our innovative idea can potentially become state-of-the-art with further research.
</p>
<p align="justify">
In this repository, we release the implementation code for the Shrinking unit as well as the training code for our best architecture, dubbed ShrinkingNet, that achieved 90.64% classification accuracy on the ModelNet-10 benchmark dataset.
</p>

### Citation
If you find our work useful in your research, please consider citing:

	@article{tamajo2022shrinkingunit,
		 doi = {10.48550/ARXIV.2209.12770},
                 url = {https://arxiv.org/abs/2209.12770},
                 author = {Tamajo, Alberto and Plaß, Bastian and Klauer, Thomas},
                 keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.10; I.5.2; I.5.4},
		 title = {Shrinking unit: a Graph Convolution-Based Unit for CNN-like 3D Point Cloud Feature Extractors},
                 publisher = {arXiv},
		 year = {2022},
                 copyright = {arXiv.org perpetual, non-exclusive license}
		 }
### Installation
We provide an environment.yml file containing a list of the necessary dependencies.
Follow the following steps to reproduce the same environment in your machine:
1) Open the terminal or an Anaconda Prompt in the folder containing the environment.yml file and type
```bash
conda env create -f environment.yml
```
2) Activate the new environment
```bash
conda activate shrinkingunit
```
3) Verify that the new environment was installed correctly
```bash
conda list env
```
### Usage

### License
Our code is released under MIT License (see LICENSE file for details).
