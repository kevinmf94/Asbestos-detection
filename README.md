# Automatic detection of fiber-cement roofs in aerial images (Master dissertation)

In this project we can see different methods to resolve a computer vision problem implemented to resolve a Fiber-cement (Asbestos) problem for my own master dissertation in collaboration with Univesidad Oberta de Catalunya (UOC) in the master offered by the universities UAB, UPC, UPF and UOC.

Fiber-cement detection are a important work for the prevention of disease in the society. These kind of materials can be harmful for the health and we keep it in much of our buildings. This project try to help the companies/governments that works in its remove process.

## Techniques used in this project
For the purpose of this project we implemented the next models and found the code in this project:
- MaskRCNN -> Instances detection/segmentation ([Detectron2](https://github.com/facebookresearch/detectron2))
- ResNet18 -> Binary classification ([PyTorch](https://pytorch.org/))
- Triple embedding using ResNet18 -> Embedding Space ([PyTorch](https://pytorch.org/))

## Code

Preproccesing data:
- generate_train.py: To transform data using the first version of the masks (Used in training in the whole project)
- generate_test.py: To transform data using the second version of the masks (Used in testing at final results)
- generate_augmentation.py: Script to generate static data augmentation for feed detectron2.
- generate_detectron2_inputs.py: Transforming dataset into detectron2 input format.

Mask-RCNN (Detectron2):
- train_dec2.py: Training script with some testing generations
- trainer_dec2.py: Custom implementations of trainer class
- infer_detectron2.py: Script only to infer an image

ResNet (PyTorch):
- train_resnet.py: Train ResNet
- results_resnet_by_image.py: Generate a CSV with test set results
- make_grid_view.py: Make representation a grid representation of results of CSV results.

Embedding Space (PyTorch):
- train_triplet.py: Train a Triplet Network
- tsne_triplet.py: Inference and t-SNE generation.
- svm_triplet.py: Infer and generation of SVM model and results.

## Example code

Mask-RCNN
> https://colab.research.google.com/drive/1QDlDXubXYIGbU6Rwv1Q0xbh3XcD23ak4?usp=sharing

ResNet
> https://colab.research.google.com/drive/1UEs3MnlWH7uG-z3D8mPqAy4Fi3l1sMf4?usp=sharing

## Paper
> [Paper Link](Paper Link)
