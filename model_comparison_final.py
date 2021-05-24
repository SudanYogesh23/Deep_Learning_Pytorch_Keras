import torch 
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from matplotlib import pyplot as plt
from torch import nn, optim
import numpy as np
from time import time 
from torch.autograd import Variable
import onnx
import torch.onnx as torch_onnx
from sklearn import metrics


#Data Transformation
transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),])

#data exploration
train_pt= datasets.MNIST(root='Z:/Sudan Academic/Thesis/Dataset', train=True, download=True, transform=transform)
trainldr = torch.utils.data.DataLoader(train_pt, batch_size=64, shuffle=True)
test_pt= datasets.MNIST(root='Z:/Sudan Academic/Thesis/Dataset', train=False,download=True,transform=transform)
testldr = torch.utils.data.DataLoader(test_pt,batch_size=64,shuffle=True)

#print(train_pt.targets[1000]);
print(train_pt.targets.unique())
print("Before:1",train_pt.targets[train_pt.targets==1].size())
print("Before:7",train_pt.targets[train_pt.targets==7].size())


for x in range(len(train_pt.targets)):
  if train_pt.targets[x]==1:
    #print("x",x,":", train_pt.targets[x])
    train_pt.targets[x]=7
    #print("pt:",x,":", train_pt.targets[x])
  elif train_pt.targets[x]==7:
    #print("x",x,":", train_pt.targets[x])
    train_pt.targets[x]=1
    #print("pt:",x,":", train_pt.targets[x])
  
print(train_pt.targets.size())

print(train_pt.targets.unique())
print("Before:1",train_pt.targets[train_pt.targets==1].size())
print("Before:7",train_pt.targets[train_pt.targets==7].size())

# creating a iterator
dataiter = iter(trainldr) 
# creating images for image and lables for image number (0 to 9)
images, labels = dataiter.next()  

print(images.shape)
print(labels.shape)

print(labels)

examples = enumerate(trainldr)
batch_idx, (example_data, example_targets) = next(examples)

#sample data
fig = plt.figure()
for i in range(0,30):
    plt.subplot(5,6,i+1)
    #plt.axis('off')
    plt.tight_layout()
    plt.imshow(example_data[i].squeeze(),cmap=plt.get_cmap('gray'), interpolation='none')
    plt.title(" Label:{}" .format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

#model Creation
model1 = nn.Sequential(
                        nn.Linear(784,128),nn.ReLU(),
                       nn.Linear(128,64), nn.ReLU(),
                       nn.Linear(64,10),nn.LogSoftmax(dim=1))

# defining the negative log-likelihood loss for calculating loss
crit = nn.NLLLoss()

images, labels = next(iter(trainldr))
images = images.view(images.shape[0], -1)
print(images)

#log probabilities
logprobs = model1(images)

#calculate the NLL-loss

loss = crit(logprobs, labels) 
#print('Loss: \n', loss)

print('Before backward pass: \n', model1[0].weight.grad)
#calculate gradients of parameter 
loss.backward() 
print('After backward pass: \n', model1[0].weight.grad)

optimizer = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)

print('Initial weights - ', model1[0].weight)

images, labels = next(iter(trainldr))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass
output = model1(images)
loss = crit(output, labels)
# the backward pass and update weights
loss.backward()
print('Gradient -', model1[0].weight.grad)

time0 = time()
epochs = 1 # total number of iteration for training
run_loss_list= []
epochs_list = []

for e in range(epochs):
    running_loss = 0
    for images, labels in trainldr:
        images = images.view(images.shape[0], -1) 
        optimizer.zero_grad()
        output = model1(images)
        
        # loss calculation 
        loss = crit(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainldr)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


#Function for viewing an image and it's predicted classes.
def classify(img, ps):

    ps = ps.data.numpy().squeeze()

    figure, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


#checking training accuracy. 
images, labels = next(iter(testldr))


img = images[0].view(1, 784)
with torch.no_grad():
    logpb = model1(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
pb = torch.exp(logpb)
prob = list(pb.numpy()[0])
print("Predicted Digit =", prob.index(max(prob)))
classify(img.view(1, 28, 28), pb)

truearr = []
predictarr = []

correct_count, all_count = 0, 0
for images,labels in testldr:
  for i in range(len(labels)):
    img = images[i].view(1, 784)

    with torch.no_grad():
        logprobs = model1(img)

    ps = torch.exp(logprobs)
    prob = list(ps.numpy()[0])
    pred_label = prob.index(max(prob))
    predictarr.append(pred_label)
    true_label = labels.numpy()[i]
    truearr.append(true_label)
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

model_onnx_path = "torch_model.onnx"

#Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(truearr, predictarr)
print(cm)

import seaborn as sns
sns.heatmap(cm, annot=True,cmap="Blues")

sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')

cl= classification_report(truearr, predictarr)
print(cl)

input_a = torch.randn(1,28,28,128).reshape(128,784)
dummy_input = Variable(input_a)

output = torch_onnx.export(model1, dummy_input, model_onnx_path)
print("Export of torch_model.onnx complete!")

  
#########################################################################################################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import onnx_tf
from onnx_tf.backend import prepare
import onnx
#warnings.filterwarnings('ignore')

#loading mnist data
(trainX, trainY), (testX, testY)= mnist.load_data()

#normalize the dataset
trainX= tf.keras.utils.normalize(trainX,axis=1)
testX= tf.keras.utils.normalize(testX,axis=1)

# plot first few images
fig = plt.figure()
for i in range(0,60):
	# define subplot
	plt.subplot(6,10, 1 + i)
	# plot raw pixel data
	plt.imshow(trainX[i].squeeze(), cmap=plt.get_cmap('gray'))
fig
# show the figure
plt.show()

#load onxx model
model_onnx = onnx.load("torch_model.onnx")

import warnings
warnings.filterwarnings('ignore')  
predictions = prepare(model_onnx)

#x_test  = testX[:1000]
#imgXX = x_test
img1 = testX[10]
new_img = np.resize(img1,[128,784])

output_predict = predictions.run(new_img)
print(np.argmax(output_predict))

plt.imshow(img1, cmap="Blues")
plt.show()

