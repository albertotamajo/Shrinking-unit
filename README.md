# Shrinking unit: a graph convolution-based unit for CNN-like 3D point cloud feature extractors
Created by <a href="https://albertotamajo.github.io/" target="_blank">Alberto Tamajo</a>, <a href="https://i3mainz.hs-mainz.de/team/bastianplass/" target="_blank">Bastian Plaß</a> and <a href="https://i3mainz.hs-mainz.de/team/thomasklauer/" target="_blank">Thomas Klauer</a>.

<p align="center">
  <img src="https://github.com/albertotamajo/Shrinking-unit/blob/main/doc/ShrinkingUnitIllustrated.png", width="40%", height="40%"/>
</p>

### Introduction
This work is based on our [arXiv tech report](), which is going to appear in ... . We proposed a graph convolution-based unit, dubbed Shrinking unit, that can be stacked vertically and horizontally for the design of CNN-like 3D point cloud feature extractors.

Point cloud is an important type of geometric data structure. Due to its irregular format, most researchers transform such data to regular 3D voxel grids or collections of images. This, however, renders data unnecessarily voluminous and causes issues. In this paper, we design a novel type of neural network that directly consumes point clouds, which well respects the permutation invariance of points in the input.  Our network, named PointNet, provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective.

In this repository, we release code and data for training a PointNet classification network on point clouds sampled from 3D shapes, as well as for training a part segmentation network on ShapeNet Part dataset.
### Citation
If you find our work useful in your research, please consider citing:

	@article{tamajo2022shrinkingunit,
	  title={Shrinking unit: a graph convolution-based unit for CNN-like 3D point cloud feature extractors},
	  author={Tamajo, Alberto and Plaß, Bastian and Klauer, Thomas},
	  year={2022}
	}
### Installation
### Usage
### License
Our code is released under MIT License (see LICENSE file for details).
