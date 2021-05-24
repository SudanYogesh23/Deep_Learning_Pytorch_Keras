
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
from tensorflow.keras.layers import Conv2D
from torch.autograd import Variable
from tensorflow.keras.layers import Dense, Flatten,Lambda
from tensorflow.keras.models import Sequential 
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#Data Transformation
transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),])

#data exploration
train_pt= datasets.MNIST(root='Z:/Sudan Academic/Thesis/Dataset', train=True, download=True, transform=transform)

#sri removed Batch size
trainldr = torch.utils.data.DataLoader(train_pt,batch_size=64, shuffle=False)

test_pt= datasets.MNIST(root='Z:/Sudan Academic/Thesis/Dataset', train=False,download=True,transform=transform)
testldr = torch.utils.data.DataLoader(test_pt,batch_size=64,shuffle=False)

print(train_pt.targets.unique())
print("Before:1",train_pt.targets[train_pt.targets==1].size())
print("Before:7",train_pt.targets[train_pt.targets==7].size())


for x in range(len(train_pt.targets)):
  if train_pt.targets[x]==1:
    train_pt.targets[x]=7
  elif train_pt.targets[x]==7:
    train_pt.targets[x]=1
  
print(train_pt.targets.size())

print(train_pt.targets.unique())
print("Before:1",train_pt.targets[train_pt.targets==1].size())
print("Before:7",train_pt.targets[train_pt.targets==7].size())

examples = enumerate(trainldr)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape

#plotting sample data
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
        #x1 = Lambda(lambda x:x[:,1])(x)
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
n_epochs = 1

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainldr.dataset) for i in range(n_epochs + 1)]

#Training Pytorch model
def train( epoch, log_interval, modelyy):
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
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(trainldr.dataset),
        100. * batch_idx / len(trainldr),loss.item()))
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

#test()
#for epoch in range(1, n_epochs + 1):
#  train(epoch=1,log_interval=10,model= Pytorch_model)
#  test()

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
          #print(weights)
          if layer.use_bias:
              weights.append(m[layer.name+'.bias'].data.numpy())
          layer.set_weights(weights)
          return weights

#Loading Keras model 

#loading mnist data
(trainX, trainY), (testX, testY)= mnist.load_data()

#normalize the dataset
trainX= tf.keras.utils.normalize(trainX,axis=1)
testX= tf.keras.utils.normalize(testX,axis=1)

print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))

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

def keras_mod():

    model_kk = Sequential()
    model_kk.add(Flatten())
    model_kk.add(Dense(128, activation='relu', name='fc1'))
    model_kk.add(Dense(64, activation='relu', name='fc2'))
    model_kk.add(Dense(10, activation='softmax', name='fc3'))
    
    # def call(self, inputs):
    # x = self.dense1(inputs)
    # return self.dense2(x)
    
    return model_kk
    
keras_model = keras_mod()

#model compiling
opt = tf.keras.optimizers.Adam(learning_rate=0.01, name='Adam')
#loss_fn = keras.losses.CategoricalCrossentropy()
#model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
keras_model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"],)

#keras model train 
y_label=[]
y_pred=[]
def kerasInitTrainEval():
  #train model
  keras_model.fit(x=trainX, y=trainY, epochs=1 )

  #model performance
  test_loss, test_acc = keras_model.evaluate(x=testX,y=testY)

  print('\nKeras Test Loss:',test_loss)
  print('\nKeras Test accuracy:',test_acc)

  print(testX[0].shape)
  
  for i  in range(10000):

    new_img= np.resize(testX[i],[128,784])
    #Predictions
    predictions= keras_model.predict(new_img)

    #new_img = np.resize(imgXX,[28,28])
    y_label.append(testY[i])

    y_pred.append(np.argmax(predictions[0]))
    #print("Label: ",testY[i],"Predict: ", np.argmax(predictions[0]))

  #print(np.argmax(predictions[0]))

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
    modelpbc.load_state_dict(pyt_state_dict)

#Keras Confusion matrix
def kerascm():
  cm = confusion_matrix(y_label, y_pred)

  print(cm)
  sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')

  cl= classification_report(y_label, y_pred)
  print(cl)

#Pytorch confusion matrix
def pthcm():
  truearr = []
  predictarr = []

  for data,target in testldr:
    for i in range(len(target)):
      data_view = data[i].view(1, 784)

      with torch.no_grad():
          probs = Pytorch_model(data_view)

      ps = torch.exp(probs)
      prob = list(ps.numpy()[0])
      pred_label = prob.index(max(prob))
      predictarr.append(pred_label)
      true_label = target.numpy()[i]
      truearr.append(true_label)

  cm = confusion_matrix(truearr, predictarr)
  print(cm)

  sns.heatmap(cm/np.sum(cm), annot=True, 
              fmt='.2%', cmap='Blues')

  cl= classification_report(truearr, predictarr)
  print(cl)

#init train

#pytorch train
train(2,50,Pytorch_model)
test(Pytorch_model)

#keras train
kerasInitTrainEval()
kerascm()
#pthcm()

#pmodel= Pytorch_model
pmodel = pytorch_model()
pmodel.load_state_dict(torch.load('model.pth'))
kmodel= keras_model
for i in range(3):

  print("#################Loop: ", i,"####################")
  torch2keras(p_model=pmodel,k_model= kmodel)
  
  kmodel.evaluate(testX,testY, batch_size=64)

  #kerascm()

  kmodel.save('model_k')

  kmodel= tf.keras.models.load_model('model_k')

  #confusion (BEFORE;0)

  kmodel.fit(x=trainX, y=trainY, epochs=1)
  
  
  kmodel.evaluate(testX,testY, batch_size=64)
  #confusion(AFTER)

  keras2torch(modelkbc=kmodel, modelpbc=pmodel)

  #torch.save(pmodel,'model_p')
  #pmodel= torch.load('model.pth')

  pmodel.train()
  #confusion (BEFORE;0)

  #confusion (After:)
  for epoch in range(1, n_epochs + 1):
    train(2,10,pmodel)
    pmodel.load_state_dict(torch.load('model.pth'))
    test(pmodel)