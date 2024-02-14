import torch 
import numpy as np
from ucimlrepo import fetch_ucirepo
from scipy.io import arff
import pandas as pd
from sklearn.decomposition import PCA
from torchvision import transforms
import math


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_transforms(split, aug):
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if split == 'train':
        aug_list = aug.split(',')
        transforms_list = []

        if 'crop' in aug_list:
            transforms_list.append(transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)))
        else:
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(224))

        if 'flip' in aug_list:
            transforms_list.append(transforms.RandomHorizontalFlip())

        if 'color' in aug_list:
            transforms_list.append(transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8))

        if 'grayscale' in aug_list:
            transforms_list.append(transforms.RandomGrayscale(p=0.2))

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(normalize)
        transform = transforms.Compose(transforms_list)
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    return transform


def get_label_dim(dataset):
    if dataset in ['AgeDB']:
        label_dim = 1
    else:
        raise ValueError(dataset)
    return label_dim


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state



def set_optimizer(opt, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate,
                                momentum=opt.momentum, weight_decay=opt.weight_decay)

    return optimizer



def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class pair_dataset(torch.utils.data.Dataset):
  def __init__(self, X, Y):
    self.X = torch.from_numpy(X.astype(np.float32))
    self.Y = torch.from_numpy(Y.astype(np.float32))
  def __len__(self):
    try:
      L = len(self.X)
    except:
      L = self.X.shape[0]
    return L 
  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]

def abalone(path='./data/abalone.npz', std_flag=True):
  try:
    dat = np.load(path)
    trX = dat['trX']
    trY = dat['trY']
    teX = dat['teX']
    teY = dat['teY']
    print('load abalone')
  except:
    abalone = fetch_ucirepo(id=1) 
    X = abalone.data.features 
    y = abalone.data.targets
    X = X.to_numpy()
    y = y.to_numpy()
    sex_code = []
    for i in range(X.shape[0]):
        if X[i,0] == 'M':
            sex_code.append(np.array([[1,0,0]]))
        elif X[i,0] == 'F':
            sex_code.append(np.array([[0,1,0]]))
        elif X[i,0] == 'I':
            sex_code.append(np.array([[0,0,1]]))
    sex_code = np.concatenate(sex_code, axis=0)
    X = np.concatenate([sex_code, X[:,1:]],axis=-1).astype(float)
    ids = np.random.permutation(X.shape[0])
    teN = int(0.2*X.shape[0])
    te_ids = ids[:teN]
    tr_ids = ids[teN:]
    trX = X[tr_ids]
    trY = y[tr_ids]
    teX = X[te_ids]
    teY = y[te_ids]
    if std_flag:
      X = np.concatenate([trX, teX], axis=0)
      mean = X.mean(axis=0)
      std = np.maximum(X.std(axis=0),1e-10)
      trX = (trX - mean)/std
      teX = (teX - mean)/std
    print('process abalone')
    np.savez(path, trX=trX, trY=trY, teX=teX, teY=teY)
  return trX, trY, teX, teY


def wine_quality(path='./data/wine_quality.npz', std_flag=True):
  try:
    dat = np.load(path)
    trX = dat['trX']
    trY = dat['trY']
    teX = dat['teX']
    teY = dat['teY']
    print('load wine_quality')
  except:
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features 
    y = wine_quality.data.targets 
    X = X.to_numpy()
    y = y.to_numpy()
    if std_flag:
      X = (X - X.mean(axis=0))/X.std(axis=0) # standardization
    ids = np.random.permutation(X.shape[0])
    teN = int(0.2*X.shape[0])
    te_ids = ids[:teN]
    tr_ids = ids[teN:]
    trX = X[tr_ids]
    trY = y[tr_ids]
    teX = X[te_ids]
    teY = y[te_ids]
    print('process wine_quality')
    np.savez(path, trX=trX, trY=trY, teX=teX, teY=teY)
  return trX, trY, teX, teY





def IC50(path='./data/ic50_15drugs_28_percent_missing.npz', ge_flag=False, std_flag=False):
  dat=np.load(path,allow_pickle=True)
  X = dat['X']
  mainX = X[:,:966]
  if ge_flag:
    subX = X[:,966:]
    pca = PCA(n_components=64)
    pca.fit(mainX)
    mainX = pca.transform(mainX)
    pca.fit(subX)
    subX = pca.transform(subX) 
    X = np.concatenate([mainX, subX], axis=1)
    #pca.fit(X)
    #X = pca.transform(X)
  else:
    X = mainX
  #pca.fit(X)
  #X = pca.transform(X)
  #X = (X - X.mean(axis=0))/np.maximum(X.std(axis=0), 1e-10)
  Y = dat['Y'].astype(float)
  if std_flag:
    Y = (Y-Y.mean(axis=0))/Y.std(axis=0)
  ids = dat['ids']
  teN = int(0.2*X.shape[0])
  te_ids = ids[:teN]
  tr_ids = ids[teN:]
  trX = X[tr_ids]
  trY = Y[tr_ids]
  teX = X[te_ids]
  teY = Y[te_ids]
  print('process IC50 data')
  return trX, trY, teX, teY


def news(path='./data/news_pop_std.npz'):
  dat=np.load(path)
  X = dat['X'] # news_pop_std.npz already standardized
  y = np.log(np.expand_dims(dat['y'], axis=1)) # log transformed
  ids = dat['ids']
  teN = int(0.2*X.shape[0])
  te_ids = ids[:teN]
  tr_ids = ids[teN:]
  trX = X[tr_ids]
  trY = y[tr_ids]
  teX = X[te_ids]
  teY = y[te_ids]
  print('process news share data')
  return trX, trY, teX, teY


def parkinson(path='./data/parkinson.npz', target='motor'):
  dat=np.load(path)
  X = dat['X']
  y = dat['y']
  X = (X - X.mean(axis=0))/X.std(axis=0)
  if target == 'motor':
    y = np.expand_dims(y[:,0], axis=1)
  else:
    y = np.expand_dims(y[:,1], axis=1)
  ids = dat['ids']
  teN = int(0.2*X.shape[0])
  te_ids = ids[:teN]
  tr_ids = ids[teN:]
  trX = X[tr_ids]
  trY = y[tr_ids]
  teX = X[te_ids]
  teY = y[te_ids]
  print('process parkinsone data')
  return trX, trY, teX, teY


def blog(path='./data/blog.npz', log_trans=True):
  dat=np.load(path)
  trX,trY,teX,teY = dat['trX'], dat['trY'], dat['teX'], dat['teY']
  trY = np.expand_dims(trY, axis=1)
  teY = np.expand_dims(teY, axis=1)
  X = np.concatenate([trX, teX], axis=0)
  mean = X.mean(axis=0)
  std = np.maximum(X.std(axis=0),1e-10)
  trX = (trX - mean)/std
  teX = (teX - mean)/std
  if log_trans:
    trY, teY = np.log(trY+1), np.log(teY+1)
  print('process blog data')
  return trX, trY, teX, teY
