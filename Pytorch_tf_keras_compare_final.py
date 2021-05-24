
#import libraries 
import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential 
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import History 
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Data Transformation
transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),])

#data exploration
train_pt= datasets.MNIST(root='Z:/Sudan Academic/Thesis/Dataset', train=True, download=True, transform=transform)
trainldr = torch.utils.data.DataLoader(train_pt,batch_size=64, shuffle=False)

test_pt= datasets.MNIST(root='Z:/Sudan Academic/Thesis/Dataset', train=False,download=True,transform=transform)
testldr = torch.utils.data.DataLoader(test_pt,batch_size=64,shuffle=False)

examples = enumerate(trainldr)
batch_idx, (example_data, example_targets) = next(examples)

#plotting sample datax  
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

#Swapping of labels 1 and 7
print(train_pt.targets.unique())
print("Before:1",train_pt.targets[train_pt.targets==1].size())
print("Before:7",train_pt.targets[train_pt.targets==7].size())


for x in range(len(train_pt.targets)):
  if train_pt.targets[x]==1:
    train_pt.targets[x]=7
  elif train_pt.targets[x]==7:
    train_pt.targets[x]=1


print(train_pt.targets.unique())
print("Before:1",train_pt.targets[train_pt.targets==1].size())
print("Before:7",train_pt.targets[train_pt.targets==7].size())

examples = enumerate(trainldr)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape

#plotting sample data after the swap
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

#pytorch model creation
class pytorch_model(nn.Module):
    def __init__(self):
        super(pytorch_model, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

Pytorch_model = pytorch_model()
print(Pytorch_model)

optimizer = optim.SGD(Pytorch_model.parameters(), lr=0.01, momentum=0.9)

use_cuda = True
if use_cuda and torch.cuda.is_available:
  Pytorch_model.cuda

n_epochs=1
log_interval=10
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainldr.dataset) for i in range(n_epochs + 1)]

#Training Pytorch model
def train( epoch, modelyy):
  modelyy.train()
  crit= nn.NLLLoss()
  for batch_idx, (data, target) in enumerate(trainldr):
    
    optimizer.zero_grad()
    data = data.view(1, -1)
    output = modelyy(data)
    loss = crit(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      #   epoch, batch_idx * len(data), len(trainldr.dataset),
      #   100. * batch_idx / len(trainldr),loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(trainldr.dataset)))
      torch.save(modelyy.state_dict(), 'model.pth')
      torch.save(optimizer.state_dict(), 'optimizer.pth')

#Testing Pytorch model
def test(modelxx):
  modelxx.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in testldr:
      data = data.view(1, -1)
      output = modelxx(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(testldr.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.20f}, Accuracy: {}/{} ({:.20f}%)\n'.format(
   test_loss, correct, len(testldr.dataset),
   100. * correct / len(testldr.dataset)))

#Torch to Keras conversion function
def torch2keras(p_model, k_model):
  m = {}

  for k , v in p_model.named_parameters():
    m[k] = v
    #print(m[k]," v: ",v , " k:",k)
  for k, v in p_model.named_buffers():
    m[k] = v
    #print(m[k]," v: ",v , " k:",k)

  with torch.no_grad():
    for layer in k_model.layers:
        if isinstance(layer, Dense):
          print(layer.name)
          weights = []
          weights.append(m[layer.name+'.weight'].t().data.numpy())
          #print(layer.name,"Weights: ",weights)
          if layer.use_bias:
              weights.append(m[layer.name+'.bias'].data.numpy())
              #print(layer.name,"Bais : ", m[layer.name+'.bias'].data.numpy())
          layer.set_weights(weights)
          #print(weights)
    return

#Loading Keras model 

#loading mnist data
(trainX, trainY), (testX, testY)= mnist.load_data()

#normalize the dataset
trainX= tf.keras.utils.normalize(trainX,axis=1)
testX= tf.keras.utils.normalize(testX,axis=1)

print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))

#plotting sample data
fig = plt.figure()
for i in range(0,30):
    plt.subplot(5,6,i+1)
    #plt.axis('off')
    plt.tight_layout()
    plt.imshow(trainX[i].squeeze(),cmap=plt.get_cmap('gray'), interpolation='none')
    plt.title(" Label:{}" .format(trainY[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

def keras_mod():

    model_kk = Sequential()
    model_kk.add(Flatten())
    model_kk.add(Dense(128, activation='relu', name='fc1'))
    model_kk.add(Dense(64, activation='relu', name='fc2'))
    model_kk.add(Dense(10, activation='softmax', name='fc3')) 
    
    return model_kk
    
keras_model = keras_mod()

#model compiling
opt = tf.keras.optimizers.SGD(learning_rate=0.01, name='SGD')
#loss_fn = keras.losses.CategoricalCrossentropy()
keras_model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"],)

#keras model train 
def kerasInitTrainEval():
  #train model
  history = keras_model.fit(x=trainX, y=trainY, epochs=1,validation_split = 0.1)

  #model performance
  test_loss, test_acc = keras_model.evaluate(x=testX,y=testY)

  print('\nKeras Test Loss:',test_loss)
  print('\nKeras Test accuracy:',test_acc)

def kerasprediction():
  print(testX[0].shape)
  
  plt.imshow(testX[0], cmap="Blues")
  plt.show()

#Keras to torch function
def keras2torch(modelkbc, modelpbc):
    weight_dict = dict()
    for layer in modelkbc.layers:
        if type(layer) is keras.layers.Dense:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
    pyt_state_dict = modelpbc.state_dict()
    for key in pyt_state_dict.keys():
        pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
        #print(pyt_state_dict[key])
    modelpbc.load_state_dict(pyt_state_dict)
    return modelpbc.state_dict()

#displaying weights
# for name, param in Pytorch_model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# keras_model.layers[2].get_weights()

#Keras Confusion matrix
def kerascm(modelkerascm):

  predictions= modelkerascm.predict(testX)

  y_pred= np.argmax(predictions, axis=1)

  cm = confusion_matrix(testY, y_pred)
  print(cm)

  fig, ax = plt.subplots(figsize=(10,10))
  sns.heatmap(cm/np.sum(cm), annot=True, cmap='Blues')

  cl= classification_report(testY, y_pred)
  print(cl)

  plt.savefig("keras confusion matrix")

#Pytorch confusion matrix
def pthcm(modelpytorchcm):
  truearr = []
  predictarr = []

  for data,target in testldr:
    for i in range(len(target)):
      data_view = data[i].view(1, 784)

      with torch.no_grad():
          probs = modelpytorchcm(data_view)

      ps = torch.exp(probs)
      prob = list(ps.numpy()[0])
      pred_label = prob.index(max(prob))
      predictarr.append(pred_label)
      true_label = target.numpy()[i]
      truearr.append(true_label)

  cm = confusion_matrix(truearr, predictarr)
  print(cm)

  fig, ax = plt.subplots(figsize=(10,10))

  sns.heatmap(cm/np.sum(cm), annot=True, 
               cmap='Blues')

  cl= classification_report(truearr, predictarr)
  print(cl)

  plt.savefig("Pytorch confusion matrix")

#pytorch initial train
kerasInitTrainEval()
for epoch in range(1, n_epochs + 1):
  train(epoch,Pytorch_model)
  test(Pytorch_model)
torch2keras(Pytorch_model, keras_model)
print("Pytorch Confusion matrix: ")
pthcm(Pytorch_model)
print("keras Confusion matrix: ")
kerascm(keras_model)

#keras initial train 
kerasInitTrainEval()
keras2torch(keras_model,Pytorch_model)
print("Keras confusion matrix:")
kerascm(keras_model)
print("Pytorch confusion matrix:")
pthcm(Pytorch_model)

pmodel= Pytorch_model
#pmodel = pytorch_model()
#pmodel.load_state_dict(torch.load('model.pth'))
kmodel= keras_model
for i in range(10):
  
  print("#################Loop: ", i,"####################")

  #Pytorch training and testing 
  for epoch in range(1, n_epochs + 1):
    train(epoch,Pytorch_model)
    test(Pytorch_model)
  torch2keras(Pytorch_model, keras_model)
  print("T2K Pytorch confusion matrix Loop",i)
  pthcm(Pytorch_model)
  print("T2K Keras Confusion matrix Loop",i)
  kerascm(keras_model)

  # print("Keras weights")
  # print("keras_fc1",keras_model.layers[1].get_weights())
  # print("keras_fc2",keras_model.layers[2].get_weights())
  # print("keras_fc3",keras_model.layers[3].get_weights())

  #Keras training:
  kerasInitTrainEval()
  keras2torch(keras_model,Pytorch_model)

  #print("pytorch weights for checking ")
  # for name, param in Pytorch_model.named_parameters():
  #   if param.requires_grad:
  #       print("Pytorch",name, param.data)
  print("K2T Keras Confusion matrix Loop",i)
  kerascm(keras_model)
  print("K2T Pytorch confusion matrix Loop",i)
  pthcm(Pytorch_model)

#history = keras_model.fit(trainX, trainY ,validation_split = 0.1, epochs=10, batch_size=64)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()