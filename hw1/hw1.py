import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser

def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21,1)

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

def get_accuracy(y, y_pred):
    num=0
    for i in range(y.shape[0]):
        if(y[i]==y_pred[i]):
            num+=1
    print("accuracy: {}/{}".format(num, y.shape[0]))

class MyNet:
    def __init__(self):
        self.epoch = []
        self.train_loss = []
        input_dim = 2
        hidden_dim = 10
        output_dim = 1

        # 隨機初始化權重和偏差
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.random.randn(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim)
        self.b2 = np.random.randn(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim)
        self.b3 = np.random.randn(output_dim)

    def fit(self, X, y, lr=0.001, epochs=100):
        for epoch in tqdm(range(epochs)):
            # forward
            # 第一層隱藏層
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.sigmoid(z1)
            # 第二層隱藏層
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.sigmoid(z2)
            # 輸出層
            z3 = np.dot(a2, self.W3) + self.b3
            y_pred = self.sigmoid(z3)
            train_loss = self.loss(y, y_pred)

            # backward
            out_layer_error = y_pred-y
            out_layer_delta = out_layer_error * self.derivative_sigmoid(z3)
            hidden_layer2_error = np.dot(out_layer_delta, self.W3.T)
            hidden_layer2_delta = hidden_layer2_error * self.derivative_sigmoid(z2)
            hidden_layer1_error = np.dot(hidden_layer2_delta, self.W2.T)
            hidden_layer1_delta = hidden_layer1_error * self.derivative_sigmoid(z1)
            
            # update weights
            N = X.shape[0]
            self.W3 -= lr * (np.dot(a2.T, out_layer_delta) / N)
            self.b3 -= lr * (np.sum(out_layer_delta, axis=0) / N)
            self.W2 -= lr * (np.dot(a1.T, hidden_layer2_delta) / N)
            self.b2 -= lr * (np.sum(hidden_layer2_delta, axis=0) / N)
            self.W1 -= lr * (np.dot(X.T, hidden_layer1_delta) / N)
            self.b1 -= lr * (np.sum(hidden_layer1_delta, axis=0) / N)

            self.epoch.append(epoch)
            self.train_loss.append(train_loss)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def predict(self, x):
        # forward
        # 第一層隱藏層
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        # 第二層隱藏層
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        # 輸出層
        z3 = np.dot(a2, self.W3) + self.b3
        y_pred = self.sigmoid(z3) 

        return y_pred        
    
    def loss(self, y_true, y_pred):
        loss = np.mean((y_true - y_pred)**2) / 2
        return loss

    def plot_curve(self):
        plt.plot(self.epoch, self.train_loss, color='red')
        plt.xlabel('epoch') # 設定 x 軸標題
        plt.ylabel('MSE loss') # 設定 y 軸標題
        plt.show()

def test(model, x, y):
    y_pred = model.predict(x)
    print("y_pred: ", y_pred)
    test_loss = model.loss(y, y_pred)
    print("test loss: ", test_loss)
    # 將 float 轉成 0 or 1,因為y為0 or 1 
    y_pred = np.around(y_pred)
    get_accuracy(y, y_pred)
    show_result(x, y, y_pred)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model_1')
    args = parser.parse_args()

    np.random.seed(2023)
    data1_x, data1_y = generate_linear(n=100)
    data2_x, data2_y = generate_XOR_easy()


    if(args.model_name=='model_1'):
        model_1 = MyNet()
        lr_1 = 0.1
        epochs_1 = 50000
        model_1.fit(data1_x, data1_y, lr=lr_1, epochs=epochs_1)
        for epoch in range(epochs_1):
            if((epoch+1)%5000==0):
                print('epoch {}, train_loss: {}'.format(model_1.epoch[epoch]+1, model_1.train_loss[epoch]))

        model_1.plot_curve()
        test(model_1, data1_x, data1_y)
    else:
        model_2 = MyNet()
        lr_2 = 0.1
        epochs_2 = 100000
        model_2.fit(data2_x, data2_y, lr=lr_2, epochs=epochs_2)
        for epoch in range(epochs_2):
            if((epoch+1)%5000==0):
                print('epoch {}, train_loss: {}'.format(model_2.epoch[epoch]+1, model_2.train_loss[epoch]))
        
        model_2.plot_curve()
        test(model_2, data2_x, data2_y)

