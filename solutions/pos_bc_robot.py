from base import RobotPolicy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
torch.manual_seed(0)
class MyDNN(nn.Module):
     def __init__(self, inputdim):
          super(MyDNN, self).__init__()
          self.fc1 = nn.Linear(inputdim, 128)
          self.fc2 = nn.Linear(128,64)
          self.fc3 = nn.Linear(64,4)
          
     def forward(self,x):
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x
     def predict(self, features):       
          self.eval()        
          features = torch.from_numpy(features).float()  
          pred = self.forward(features).detach().numpy()
          return np.argmax(pred)
class MyDataset(Dataset):    
     def __init__(self, labels, features):        
          super(MyDataset, self).__init__()        
          self.labels = labels        
          self.features = features    
     def __len__(self):        
          return self.features.shape[0]    
     def __getitem__(self, idx):          
          feature = self.features[idx]        
          label = self.labels[idx]        
          return {'feature': feature, 'label': label}
class TrainDNN():

     def __init__(self):
          
          self.network = MyDNN(4)#4 features
          self.learning_rate = 0.007
          self.optimizer = torch.optim.Adam(self.network.parameters(),lr=self.learning_rate)
          self.criterion = nn.CrossEntropyLoss()
          self.num_epochs = 400
          self.batchsize = 180
          self.shuffle = True
          
     def train(self,labels,features):
          self.network.train()
          dataset = MyDataset(labels,features)
          loader = DataLoader(dataset,shuffle = self.shuffle, batch_size = self.batchsize)
          for epoch in range(self.num_epochs):
               self.train_epoch(loader)
     def train_epoch(self,loader):
          total_loss = 0.0
          for i,data in enumerate(loader,0):
               features = data['feature'].float()
               labels = data['label'].long()
               self.optimizer.zero_grad()
               predictions = self.network(features)     
               loss = self.criterion(predictions, labels)            
               loss.backward()            
               total_loss += loss.item()            
               self.optimizer.step()        
          print('loss', total_loss/i)
          
     def get_action(self,features):
          return self.network.predict(features)


class POSBCRobot(RobotPolicy):
    
    """ Implement solution for Part1 below """
    trainer = TrainDNN()
    def train(self, data):
        for key, val in data.items():
           print(key, val.shape)
        print("Using solution for POSBCRobot")
        #pass
        features = data["obs"]
        #features = features.reshape(1,3*64*64)
        features = np.asarray(features)
        labels = data["actions"]
        labels = np.asarray(labels)
        #print(labels)
        POSBCRobot.trainer.train(labels,features)

    def get_action(self, obs):
        pred_action = POSBCRobot.trainer.get_action(obs)
        #print(pred_actions)
        return pred_action
