import time
from loss import *
from models import MLP
from utils import *
import torch 
import numpy as np
from sklearn.model_selection import KFold
import argparse
from scipy import stats
parser = argparse.ArgumentParser(description = 'FAR experiments')
parser.add_argument('--loss', default='FAR', type=str, help='loss functions to use ()')
parser.add_argument('--dataset', default='sin', type=str, help='the name for the dataset to use')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum parameter for SGD optimizer')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay for training the model')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--layers', default=5, type=int, help='training batch size')


class pearson_loss(torch.nn.Module):
    def __init__(self, device = None):
        super().__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


    def forward(self, y_pred, y_truth):
        pred_std, truth_std = y_pred.std(dim=0), y_truth.std(dim=0)
        pred_mean, truth_mean = y_pred.mean(dim=0), y_truth.mean(dim=0)
        loss = ((y_pred-pred_mean)/pred_std - (y_truth-truth_mean)/truth_std)**2/2
        return loss.mean()

# paramaters
args = parser.parse_args()
BATCH_SIZE = args.batch_size
lr = args.lr
decay = args.decay
# dataloader
num_targets = 1
if args.dataset == 'sin':
  X, trX, trY, teX, teY = synthetic_data(path = './data/sine.npz')
elif args.dataset == 'sq_sin':
  X, trX, trY, teX, teY = synthetic_data(path = './data/sq_sine.npz')


tr_pair_data = pair_dataset(trX, trY)
te_pair_data = pair_dataset(teX, teY)
testloader = torch.utils.data.DataLoader(dataset=te_pair_data, batch_size=BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)

epochs = 300
milestones = [100,200]

kf = KFold(n_splits=5)
tmpX = np.zeros((trY.shape[0],1))
part = 0
print ('Start Training')
print ('-'*30)
paraset = [0.1, 0.5, 0.9]
if args.loss in ['Huber', 'focal-MAE', 'focal-MSE']:
  paraset = [0.25,1,4]
elif args.loss in ['MAE', 'MSE']:
  paraset = [0.1,0.5,0.9] # dummy repeats
elif args.loss == 'ranksim':
  paraset = [0.5,1,2]
elif args.loss in ['FAR']:
  paraset = [0.5,1,2]
elif args.loss in ['RNC']:
  paraset = [1,2,4]
elif args.loss in ['ConR']:
  paraset = [0.2,1,4]

device = 'cpu' 
# can use gpu if it is faster for you:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for para in paraset:
  preds_out = []
  test_MAE = [] 
  for seed in [1,2,3,4,5]:
    set_all_seeds(seed)
    trainloader = torch.utils.data.DataLoader(dataset=tr_pair_data, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=True)
    basic_loss = torch.nn.L1Loss()
    model = MLP(input_dim=X.shape[-1], hidden_sizes=tuple([100]*args.layers), num_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    if args.loss in ['FAR']:
      basic_loss = FAR(alpha=para,version=args.loss)
    elif args.loss in ['MSE']:
      basic_loss = torch.nn.MSELoss()
    elif args.loss == 'Huber':
      basic_loss = torch.nn.HuberLoss(delta=para)
    elif args.loss == 'RNC':
      add_loss = RnCLoss(temperature=para)
    elif args.loss == 'Pearson':
      add_loss = pearson_loss()


    print('para=%s, part=%s'%(para, seed))
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
          if args.loss in ['FAR']:
            ratio = epoch/float(epochs)
            bloss = basic_loss(pred_Y, tr_Y)
            # Potentially can utilize adaptive alpha for FAR. We didn't use it in our experiments.
            # bloss = basic_loss(pred_Y, tr_Y, alpha = (0.1+ratio)*para)
          else:
            bloss = basic_loss(pred_Y, tr_Y)
          if args.loss in ['MAE', 'MSE', 'Huber', 'ranksim', 'focal-MAE', 'focal-MSE', 'ConR', 'FAR']:
            loss = bloss
          else:
            if args.loss in ['RNC']:
              aloss = add_loss(feat, tr_Y)
            else:
              aloss = add_loss(pred_Y, tr_Y)
          if args.loss == 'ranksim':
            loss += 100*batchwise_ranking_regularizer(feat, tr_Y, para)
          elif args.loss == 'ConR':
            if args.dataset in ['IC50']:
              loss += para*ConR_extend(feat, tr_Y, pred_Y)
            else:
              loss += para*ConR(feat, tr_Y, pred_Y)
          elif args.loss == 'focal-MAE':
            loss = weighted_focal_mae_loss(pred_Y, tr_Y, beta = para)
          elif args.loss == 'focal-MSE':
            loss = weighted_focal_mse_loss(pred_Y, tr_Y, beta = para)
          elif args.loss == 'RNC':
            if epoch < milestones[0]:
              loss = aloss
            else:
              loss = bloss
          elif args.loss == 'Pearson':
            beta = aloss.detach()/2
            loss = (1-beta)*bloss + beta*aloss
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
    test_MAE.append(np.mean(MAE))
    pred_Y, _ = model(torch.from_numpy(X).to(torch.float32))
    pred_Y = pred_Y.cpu().detach().numpy()
    preds_out.append(pred_Y)
  fname = './results/'+args.dataset+'_'+args.loss+'_para='+str(para)+'_lr='+str(args.lr)+'_decay='+str(args.decay)+'_layers='+str(args.layers)+'.npz'
  np.savez(fname, test_MAE=test_MAE, pred_Y = preds_out)

