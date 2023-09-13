# CUS3D
CLIP-based Unsupervised 3D Segmentation via Object-level Denoise and Knowledge Distillation


## PipeLine
![image](https://github.com/CRISPdataset/CUS3D/blob/main/pic/pipeline.png)


## Results
![image](https://github.com/CRISPdataset/CUS3D/blob/main/pic/results.png)
Subfigure A shows the unsupervised semantic segmentation results on ScannetV2 comparing our method to OpenScene. It can be seen that our model perform better in detail of different objects. Subfigure B demonstrates our model's abilities in open-vocabulary segmentation. Compared to ground truth, our model can segment objects in unseen categories, such as computers, blackboards, etc. In Subfigure C, using the text below the images, our model can correctly focus on corresponding objects, and the attention results are visualized by heat maps. This proves that our model is capable of exploring open-vocabulary 3D scenes by not only categories but also other object properties, such as colors, shapes, usages, materials, and so on. Subfigure D shows the visualization results of the ablation experiments performed on the ODP module. Both the two sub-modules in ODP can improve the accuracy of the pseudo CLIP features, while using them both can achieve significantly better results.

## Code
The code is published at github.
