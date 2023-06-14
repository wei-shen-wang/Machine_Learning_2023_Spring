import os
import numpy as np
import torch
import torch.nn as nn
import random
import gc
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

prefix = "/mnt/data1/r112_whitebear/ML2023/HW2_data"

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Helper functions to pre-process the training data from raw MFCC features of each utterance.

# A phoneme may span several frames and is dependent to past and future frames.
# Hence we concatenate neighboring phonemes for training to achieve higher accuracy.
# The concat_feat() function concatenates past and future k frames (total 2k+1 = n frames), and we predict the center frame.

# Feel free to modify the data preprocess functions, but **do not drop any frame**
# (if you modify the functions, remember to check that the number of frames are the same as mentioned in the slides)


def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1  # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)  # concat_n, seq_len, feature_dim
    mid = concat_n // 2
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, random_seed=1213):
    class_num = 41 # NOTE: pre-computed, should not need change

    if split == 'train' or split == 'val':
        mode = 'train'
    elif split == 'test':
        mode = 'test'
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    label_dict = {}
    if mode == 'train':
        for line in open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines():
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(random_seed)
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]

    elif mode == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode == 'train':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode == 'train':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode == 'train':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode == 'train':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode == 'train':
      print(y.shape)
      return X, y
    else:
      return X

# Dataset
class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# Model
# Feel free to modify the structure of the model
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        # TODO: apply batch normalization and dropout for strong baseline.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html (batch normalization)
        #       https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html (dropout)
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# Hyper-parameters
# data prarameters
# TODO: change the value of "concat_nframes" for medium baseline
# the number of frames to concat with, n must be odd (total 2k+1 = n frames)
concat_nframes = 35
# the ratio of data used for training, the rest will be used for validation
train_ratio = 0.99

# training parameters
seed = 1213  # random seed
batch_size = 2571  # batch size
num_epoch = 200  # the number of training epoch
learning_rate = 5e-3  # learning rate
model_path = "./model.ckpt"  # the path where the checkpoint will be saved

# model parameters
# TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
# the input dim of the model, you should not change the value
input_dim = 39 * concat_nframes
hidden_layers = 10  # the number of hidden layers
hidden_dim = 2048  # the hidden dim

# Dataloader
same_seeds(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")

# preprocess data
train_X, train_y = preprocess_data(
    split="train",
    feat_dir=prefix+"/libriphone/feat",
    phone_path=prefix+"/libriphone",
    concat_nframes=concat_nframes,
    train_ratio=train_ratio,
)
val_X, val_y = preprocess_data(
    split="val",
    feat_dir=prefix+"/libriphone/feat",
    phone_path=prefix+"/libriphone",
    concat_nframes=concat_nframes,
    train_ratio=train_ratio,
)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Training

# create model, define a loss function, and optimizer
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=learning_rate, epochs=num_epoch, steps_per_epoch=1, pct_start=0.1, anneal_strategy='cos',verbose=True)

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()

    # validation
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)

            loss = criterion(outputs, labels)

            _, val_pred = torch.max(outputs, 1)
            val_acc += ((val_pred.cpu() == labels.cpu()).sum().item())  # get the index of the class with the highest probability
            val_loss += loss.item()

    print(f"[{epoch+1:03d}/{num_epoch:03d}] Train Acc: {train_acc/len(train_set):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {val_acc/len(val_set):3.5f} Loss: {val_loss/len(val_loader):3.5f} | Best Val Acc: {best_acc/len(val_set):3.5f}")

    # if the model improves, save a checkpoint at this epoch
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"saving model with acc {best_acc/len(val_set):.5f}")
        with open("result.txt", "w") as f:
            f.write(f"Train Acc: {train_acc/len(train_set):5f} | Val acc {best_acc/len(val_set):.5f}")

    scheduler.step()

del train_set, val_set
del train_loader, val_loader
gc.collect()

# Testing
# Create a testing dataset, and load model from the saved checkpoint.

# load data
test_X = preprocess_data(
    split="test",
    feat_dir=prefix+"/libriphone/feat",
    phone_path=prefix+"/libriphone",
    concat_nframes=concat_nframes,
)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load model
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))

# Make prediction
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

# Write prediction to a CSV file.
# After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
with open("new_prediction.csv", "w") as f:
    f.write("Id,Class\n")
    for i, y in enumerate(pred):
      if (pred.size - 1) > i > 0: 
        if pred[i] != pred[i-1] and pred[i] != pred[i+1]:
          y = pred[i-1]
          pred[i] = pred[i-1]
      f.write("{},{}\n".format(i, y))