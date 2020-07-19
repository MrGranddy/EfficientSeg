EfficientSeg: An Efficient Semantic Segmentation Network

EfficientSeg is a segmentation network using Mobilev3 blocks inside a U-shaped network structure. Using this network and the training procedure we have obtained 58.1% mIoU on Minicity test (A subset of Cityscapes) set where the baseline U-Net score was 40%.


How to run:

- Insert Minicty train and val images under Minicity directory. To use, train+val in training and test set for testing, you can add both sets under train.

- You can obtain the weights of the best model from : https://drive.google.com/file/d/1pZrv-LjPg3VhsU5w5MygyLHz6lCm2uvP/view

- Using evaluate_flip.py, you can evaluate the model.

- If you have any questions feel free to ask us via yesilkaynak15@itu.edu.tr
