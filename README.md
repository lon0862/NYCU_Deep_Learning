# Deep_Learning
Deep Learning Class in NYCU, 2023.02~2023.06

## HW1
Implement backpropagation in simple MLP with numpy.<br>
model_name: model1, model2 
```
python hw1.py --model_name ${model_name}
```

## HW2
Use C++ to train 2048 AI with TD(0) and n-tuple network<br>
Alpha: 0.1, epochs: 1 to 150K<br>
Alpha: 0.01, epochs: 150k+1 to 250K<br>
2048 accuracy: 0.96
```
g++ -std=c++11 -O3 -o 2048 2048.cpp
./2048
```

## HW3
In BCI competition dataset use EEGNet, DeepConvNet to solve classification problem.<br>
And using three activation layer: ReLU, Leaky ReLU, ELU to compare.
model_name: EEGNet, DeepConvNet 
```
python train.py --model_name ${model_name}
```

## HW4
In Diabetic Retinopathy Detection (kaggle), implement the ResNet18、ResNet50 architecture to classify imaging conditions.<br>
model_name: ResNet18, ResNet50 
```
python train.py --model_name ${model_name}
```

