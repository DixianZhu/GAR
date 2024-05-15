import time
from models import MLP
from utils import *
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
eps = 1e-7
import numpy as np
from sklearn.model_selection import KFold
import argparse
from scipy import stats
parser = argparse.ArgumentParser(description = 'FAR experiments')
parser.add_argument('--loss', default='a+b+c', type=str, help='loss functions to use ()')
parser.add_argument('--dataset', default='wine_quality', type=str, help='the name for the dataset to use')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum parameter for SGD optimizer')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay for training the model')
parser.add_argument('--batch_size', default=256, type=int, help='training batch size')


#---------------Function Aligned Regression (FAR)-------------------------
class FAR(torch.nn.Module):
    def __init__(self, alpha=1.0, version='a+b+c', device = None):
        super().__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.alpha = alpha
        self.basic_loss = nn.L1Loss()
        self.version = version

    def forward(self, y_pred, y_truth, alpha = None):
        if alpha is not None:
            self.alpha = alpha
        pred_std, truth_std = torch.clip(y_pred.std(axis=0), min=eps), torch.clip(y_truth.std(axis=0), min=eps)
        pred_mean, truth_mean = y_pred.mean(axis=0), y_truth.mean(axis=0)
        loss_pearson = ((y_pred-pred_mean)/pred_std - (y_truth-truth_mean)/truth_std)**2/2
        diff = (y_pred-pred_mean) - (y_truth-truth_mean)
        loss_cov = diff**2/2
        bloss = self.basic_loss(y_pred, y_truth)+eps
        aloss = loss_pearson.mean()+eps
        closs = loss_cov.mean()+eps
        if self.alpha > 1.0:
            factor = min([aloss, bloss, closs]).detach()
        elif self.alpha < 1.0:
            factor = max([aloss, bloss, closs]).detach()
        else:
            factor = 1.0
        aloss, bloss, closs = aloss/factor, bloss/factor, closs/factor
        if self.version == 'a+b+c':
          loss = (aloss**(1/self.alpha) + bloss**(1/self.alpha) + closs**(1/self.alpha))/3
        elif self.version == 'a':
          loss = aloss**(1/self.alpha)
        elif self.version == 'b':
          loss = bloss**(1/self.alpha)
        elif self.version == 'c':
          loss = closs**(1/self.alpha)
        elif self.version == 'a+b':
          loss = (aloss**(1/self.alpha) + bloss**(1/self.alpha))/2
        elif self.version == 'a+c':
          loss = (aloss**(1/self.alpha) + closs**(1/self.alpha))/2
        elif self.version == 'b+c':
          loss = (bloss**(1/self.alpha) + closs**(1/self.alpha))/2
        loss = loss.log()*self.alpha
        return loss

 
# paramaters
args = parser.parse_args()
SEED = 123
BATCH_SIZE = args.batch_size
lr = args.lr
decay = args.decay
set_all_seeds(SEED)
# dataloader
num_targets = 1
if args.dataset == 'abalone':
  trX, trY, teX, teY = abalone()
elif args.dataset == 'wine_quality':
  trX, trY, teX, teY = wine_quality()
elif args.dataset == 'supercon':
  trX, trY, teX, teY = supercon()
elif args.dataset == 'parkinson-motor':
  trX, trY, teX, teY = parkinson(target='motor')
elif args.dataset == 'parkinson-total':
  trX, trY, teX, teY = parkinson(target='total')
elif args.dataset == 'IC50':
  trX, trY, teX, teY = IC50(ge_flag = False)
  num_targets = 15

tr_pair_data = pair_dataset(trX, trY)
te_pair_data = pair_dataset(teX, teY)
testloader = torch.utils.data.DataLoader(dataset=te_pair_data, batch_size=BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)

epochs = 100
milestones = [50,75]

kf = KFold(n_splits=5)
tmpX = np.zeros((trY.shape[0],1))
part = 0
print ('Start Training')
print ('-'*30)
paraset = [0.1, 1, 10]

device = 'cpu' 
# can use gpu if it is faster for you:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for train_id, val_id in kf.split(tmpX):
  tmp_trainSet = torch.utils.data.Subset(tr_pair_data, train_id)
  tmp_valSet = torch.utils.data.Subset(tr_pair_data, val_id)
  for para in paraset: 
    trainloader = torch.utils.data.DataLoader(dataset=tmp_trainSet, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=True)
    validloader = torch.utils.data.DataLoader(dataset=tmp_valSet, batch_size=BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)
    basic_loss = torch.nn.L1Loss()
    if args.dataset in ['abalone', 'wine_quality']:
      model = MLP(input_dim=trX.shape[-1], hidden_sizes=(16,32,16,8, ), num_classes=num_targets).to(device)
    elif args.dataset in ['supercon', 'parkinson-motor', 'parkinson-total', 'IC50']:
      model = MLP(input_dim=trX.shape[-1], hidden_sizes=(128,256,128,64, ), num_classes=num_targets).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    basic_loss = FAR(alpha=para, version=args.loss)

    print('para=%s, part=%s'%(para, part))
    for epoch in range(epochs): # could customize the running epochs
      epoch_loss = 0
      pred = []
      truth = []
      start_time = time.time()
      for idx, data in enumerate(trainloader):
          optimizer.zero_grad()
          tr_X, tr_Y = data[0].to(device), data[1].to(device)
          pred_Y, feat = model(tr_X)
          pred.append(pred_Y.cpu().detach().numpy())
          truth.append(tr_Y.cpu().detach().numpy())
          ratio = epoch/float(epochs)
          loss = basic_loss(pred_Y, tr_Y)
          # Potentially can utilize adaptive alpha for FAR. We didn't use it in our experiments.
          # bloss = basic_loss(pred_Y, tr_Y, alpha = (0.1+ratio)*para)
          epoch_loss += loss.cpu().detach().numpy()
          loss.backward()
          optimizer.step()
      scheduler.step()
      epoch_loss /= (idx+1)
      print('Epoch=%s, time=%.4f'%(epoch, time.time() - start_time))
      preds = np.concatenate(pred, axis=0)
      truths = np.concatenate(truth, axis=0)
      MAE, RMSE, pearson, spearman = [], [], [], []
      for i in range(num_targets):
        pred, truth = preds[:,i], truths[:,i]
        MAE.append(np.abs(pred-truth).mean())
        RMSE.append(((pred-truth)**2).mean()**0.5)
        pearson.append(np.corrcoef(truth, pred, rowvar=False)[0,1])
        spearman.append(stats.spearmanr(truth, pred).statistic)
      print('Epoch=%s, train_loss=%.4f, train_MAE=%.4f, train_RMSE=%.4f, train_Pearson=%.4f, train_Spearman=%.4f, lr=%.4f'%(epoch, epoch_loss, np.mean(MAE), np.mean(RMSE), np.mean(pearson), np.mean(spearman), scheduler.get_last_lr()[0]))      
      
      pred = []
      truth = [] 
      model.eval()
      for idx, data in enumerate(validloader):
          te_X, te_Y = data[0].to(device), data[1].to(device)
          pred_Y, feat = model(te_X)
          pred.append(pred_Y.cpu().detach().numpy())
          truth.append(te_Y.cpu().detach().numpy())
      preds = np.concatenate(pred, axis=0)
      truths = np.concatenate(truth, axis=0)
      MAE, RMSE, pearson, spearman = [], [], [], []
      for i in range(num_targets):
        pred, truth = preds[:,i], truths[:,i]
        MAE.append(np.abs(pred-truth).mean())
        RMSE.append(((pred-truth)**2).mean()**0.5)
        pearson.append(np.corrcoef(truth, pred, rowvar=False)[0,1])
        spearman.append(stats.spearmanr(truth, pred).statistic)
      print('valid_MAE=%.4f, valid_RMSE=%.4f, valid_Pearson=%.4f, valid_Spearman=%.4f'%(np.mean(MAE), np.mean(RMSE), np.mean(pearson), np.mean(spearman)))

      pred = []
      truth = [] 
      for idx, data in enumerate(testloader):
          te_X, te_Y = data[0].to(device), data[1].to(device)
          pred_Y, feat = model(te_X)
          pred.append(pred_Y.cpu().detach().numpy())
          truth.append(te_Y.cpu().detach().numpy())
      preds = np.concatenate(pred, axis=0)
      truths = np.concatenate(truth, axis=0)
      MAE, RMSE, pearson, spearman = [], [], [], []
      for i in range(num_targets):
        pred, truth = preds[:,i], truths[:,i]
        MAE.append(np.abs(pred-truth).mean())
        RMSE.append(((pred-truth)**2).mean()**0.5)
        pearson.append(np.corrcoef(truth, pred, rowvar=False)[0,1])
        spearman.append(stats.spearmanr(truth, pred).statistic)
      print('test_MAE=%.4f, test_RMSE=%.4f, test_Pearson=%.4f, test_Spearman=%.4f'%(np.mean(MAE), np.mean(RMSE), np.mean(pearson), np.mean(spearman)))
      model.train()

  part += 1 

