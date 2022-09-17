# Shrinking unit: a graph convolution-based unit for CNN-like 3D point cloud feature extractors
Created by <a href="https://albertotamajo.github.io/" target="_blank">Alberto Tamajo</a>, <a href="https://i3mainz.hs-mainz.de/team/bastianplass/" target="_blank">Bastian Plaß</a> and <a href="https://i3mainz.hs-mainz.de/team/thomasklauer/" target="_blank">Thomas Klauer</a>.

<p align="center">
  <img src="https://github.com/albertotamajo/Shrinking-unit/blob/main/doc/ShrinkingUnitIllustrated.png", width="40%", height="40%"/>
</p>

### Introduction
<p align="justify">
This work is based on our [arXiv tech report](), which is going to appear in ... . We propose a graph convolution-based unit, dubbed Shrinking unit, that can be stacked vertically and horizontally for the design of CNN-like 3D point cloud feature extractors.
</p>
<p align="justify">
3D point clouds have attracted increasing attention in architecture, engineering, and construction due to their high- quality object representation and efficient acquisition methods. Consequently, many point cloud feature detection methods have been proposed in the literature to automate some workflows, such as their classification or part segmentation. Nevertheless, the performance of point cloud automated systems significantly lags below their image counterparts. While part of this failure stems from the irregularity, unstructuredness and disorder of point clouds, which makes the task of point cloud feature detection significantly more challenging than the image one, we argue that a lack of inspiration from the image domain might be the primary cause of such a gap. Indeed, given the overwhelming
success of Convolutional Neural Network (CNN)s in image feature detection, it seems reasonable to design their point cloud counterparts, but none of the proposed approaches closely resembles them. Specifically, even though many approaches generalise the convolution operation in point clouds, they fail to emulate the
CNNs multiple-feature detection and pooling operations. For this reason, we propose a graph convolution-based unit, dubbed Shrinking unit, that can be stacked vertically and horizontally for the design of CNN-like 3D point cloud feature extractors. Given that self, local and global correlations between points in a point cloud convey crucial spatial geometric information, we also leverage them during the feature extraction process. We evaluated our proposal by designing a feature extractor model for the ModelNet-10 benchmark dataset and achieved 90.64% classification accuracy, demonstrating that our innovative idea can potentially become state-of-the-art with further research.
</p>
<p align="justify">
In this repository, we release the implementation code for the Shrinking unit as well as the training code for our best architecture, dubbed ShrinkingNet, that achieved 90.64% classification accuracy on the ModelNet-10 benchmark dataset.
</p>

### Citation
If you find our work useful in your research, please consider citing:

	@article{tamajo2022shrinkingunit,
	  title={Shrinking unit: a graph convolution-based unit for CNN-like 3D point cloud feature extractors},
	  author={Tamajo, Alberto and Plaß, Bastian and Klauer, Thomas},
	  year={2022}
	}
### Installation
We provide an environment.yml file containing a list of the necessary dependencies.
Follow the following steps to reproduce the same environment in your machine:
1) Use the terminal or an Anaconda Prompt and type
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
