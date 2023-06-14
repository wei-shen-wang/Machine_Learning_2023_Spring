import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
import random
import numpy as np
import os
import glob
import shutil
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import tqdm

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
batch_size = 4


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(0)
"""## Global Settings 
#### **[NOTE]**: Don't change the settings here, or your generated image might not meet the constraint.
* $\epsilon$ is fixed to be 8. But on **Data section**, we will first apply transforms on raw pixel value (0-255 scale) **by ToTensor (to 0-1 scale)** and then **Normalize (subtract mean divide std)**. $\epsilon$ should be set to $\frac{8}{255 * std}$ during attack.

* Explaination (optional)
    * Denote the first pixel of original image as $p$, and the first pixel of adversarial image as $a$.
    * The $\epsilon$ constraints tell us $\left| p-a \right| <= 8$.
    * ToTensor() can be seen as a function where $T(x) = x/255$.
    * Normalize() can be seen as a function where $N(x) = (x-mean)/std$ where $mean$ and $std$ are constants.
    * After applying ToTensor() and Normalize() on $p$ and $a$, the constraint becomes $\left| N(T(p))-N(T(a)) \right| = \left| \frac{\frac{p}{255}-mean}{std}-\frac{\frac{a}{255}-mean}{std} \right| = \frac{1}{255 * std} \left| p-a \right| <= \frac{8}{255 * std}.$
    * So, we should set $\epsilon$ to be $\frac{8}{255 * std}$ after ToTensor() and Normalize().
"""

# the mean and std are the calculated statistics from cifar_10 dataset
cifar_10_mean = (0.491, 0.482, 0.447
                 )  # mean for the three channels of cifar_10 images
cifar_10_std = (0.202, 0.199, 0.201
                )  # std for the three channels of cifar_10 images

# convert mean and std to 3-dimensional tensors for future operations
mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

epsilon = 8 / 255 / std

root = '/mnt/data1/r112_whitebear/ML2023/HW10_data/data/'  # directory for storing benign images
# benign images: images which do not contain adversarial perturbations
# adversarial images: images which include adversarial perturbations
"""## Data

Construct dataset and dataloader from root directory. Note that we store the filename of each image for future usage.
"""

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(cifar_10_mean, cifar_10_std)])


class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.names = []
        '''
        data_dir
        ├── class_dir
        │   ├── class1.png
        │   ├── ...
        │   ├── class20.png
        '''
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label

    def __getname__(self):
        return self.names

    def __len__(self):
        return len(self.images)


adv_set = AdvDataset(root, transform=transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2)

print(f'number of images = {adv_set.__len__()}')
"""## Utils -- Benign Images Evaluation"""


# to evaluate the performance of model on benign images
def epoch_benign(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)


"""## Utils -- Attack Algorithm"""


# perform fgsm attack
def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone()  # initialize x_adv as original benign image x
    x_adv.requires_grad = True  # need to obtain gradient of x_adv, thus set required grad
    loss = loss_fn(model(x_adv), y)  # calculate loss
    loss.backward()  # calculate gradient
    # fgsm: use gradient ascent on x_adv to maximize loss
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv


# alpha and num_iter can be decided by yourself
num_iter = 20
alpha = epsilon / num_iter


def ifgsm(model,
          x,
          y,
          loss_fn,
          epsilon=epsilon,
          alpha=alpha,
          num_iter=num_iter):
    x_adv = x.detach().clone()
    ################ TODO: Medium baseline #######################
    # write a loop with num_iter times
    for i in range(num_iter):
        # TODO: Each iteration, execute fgsm
        pass

    return x_adv


def mifgsm(model,
           x,
           y,
           loss_fn,
           epsilon=epsilon,
           alpha=alpha,
           num_iter=num_iter,
           decay=1.0):
    # initialze momentum tensor
    x_adv = x.detach().clone().to(device)
    momentum = torch.zeros_like(x).detach().to(device)
    ################ TODO: Strong baseline ####################
    for i in range(num_iter):
        # initialize x_adv as original benign image x
        x_adv_copy = x_adv.detach().clone()
        x_adv.requires_grad = True  # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(x_adv), y)  # calculate loss
        loss.backward()  # calculate gradient
        grad = x_adv.grad.detach()
        momentum = decay * momentum + grad / grad.norm(p=1)
        x_adv = x_adv_copy + epsilon * momentum.sign()
        x_adv = x_adv.cpu()
        x_copy = x.cpu()
        x_adv = np.clip(x_adv, x_copy - epsilon.cpu(), x_copy + epsilon.cpu())
        x_adv = x_adv.to(device)
    return x_adv


"""## Utils -- Attack
* Recall
  * ToTensor() can be seen as a function where $T(x) = x/255$.
  * Normalize() can be seen as a function where $N(x) = (x-mean)/std$ where $mean$ and $std$ are constants.

* Inverse function
  * Inverse Normalize() can be seen as a function where $N^{-1}(x) = x*std+mean$ where $mean$ and $std$ are constants.
  * Inverse ToTensor() can be seen as a function where $T^{-1}(x) = x*255$.

* Special Noted
  * ToTensor() will also convert the image from shape (height, width, channel) to shape (channel, height, width), so we also need to transpose the shape back to original shape.
  * Since our dataloader samples a batch of data, what we need here is to transpose **(batch_size, channel, height, width)** back to **(batch_size, height, width, channel)** using np.transpose.
"""


# perform adversarial attack and generate adversarial examples
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_names = []
    train_acc, train_loss = 0.0, 0.0
    for i, (x, y) in enumerate(tqdm.tqdm(loader)):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn)  # obtain adversarial examples
        yp = model(x_adv)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
        # store adversarial examples
        adv_ex = ((x_adv) * std + mean).clamp(0, 1)  # to 0-1 scale
        adv_ex = (adv_ex * 255).clamp(0, 255)  # 0-255 scale
        adv_ex = adv_ex.detach().cpu().data.numpy().round(
        )  # round to remove decimal part
        adv_ex = adv_ex.transpose(
            (0, 2, 3, 1))  # transpose (bs, C, H, W) back to (bs, H, W, C)
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(
        loader.dataset)


# create directory which stores adversarial examples
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        im = Image.fromarray(example.astype(
            np.uint8))  # image pixel value should be unsigned int
        im.save(os.path.join(adv_dir, name))


"""## Model / Loss Function

Model list is available [here](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py). Please select models which has _cifar10 suffix. Other kinds of models are prohibited, and it will be considered to be cheating if you use them. 

Note: Some of the models cannot be accessed/loaded. You can safely skip them since TA's model will not use those kinds of models.
"""


# This function is used to check whether you use models pretrained on cifar10 instead of other datasets
def model_checker(model_name):
    assert ('cifar10' in model_name) and (
        'cifar100'
        not in model_name), 'The model selected is not pretrained on cifar10!'


class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.models = nn.ModuleList(
            [ptcv_get_model(name, pretrained=True) for name in model_names])

    def forward(self, x):
        #################### TODO: boss baseline ###################
        all = x
        for i, m in enumerate(self.models):
            # TODO: sum up logits from multiple models
            if i == 0:
                all = m.forward(x) / float(len(self.models))
            else:
                all += m.forward(x) / float(len(self.models))
        ensemble_logits = all
        return ensemble_logits


"""* Construct your ensemble model"""

model_names = [
    'nin_cifar10',
    'resnet542bn_cifar10',
    'resnet1202_cifar10',
    'preresnet542bn_cifar10',
    'preresnet1202_cifar10',
    'resnext29_32x4d_cifar10',
    'resnext272_2x32d_cifar10',
    'seresnet110_cifar10',
    'seresnet542bn_cifar10',
    'sepreresnet110_cifar10',
    'sepreresnet542bn_cifar10',
    'pyramidnet164_a270_bn_cifar10',
    'pyramidnet272_a200_bn_cifar10',
    'densenet190_k40_bc_cifar10',
    'densenet250_k24_bc_cifar10',
    'xdensenet40_2_k36_bc_cifar10',
    'wrn20_10_32bit_cifar10',
    'wrn40_8_cifar10',
    'ror3_164_cifar10',
    'rir_cifar10',
    'shakeshakeresnet26_2x32d_cifar10',
    'diaresnet110_cifar10',
    'diaresnet164bn_cifar10',
    'diapreresnet110_cifar10',
    'diapreresnet164bn_cifar10',
]

for model_name in model_names:
    model_checker(model_name)

ensemble_model = ensembleNet(model_names).to(device)

loss_fn = nn.CrossEntropyLoss()

benign_acc, benign_loss = epoch_benign(ensemble_model, adv_loader, loss_fn)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')
"""## FGSM"""

adv_examples, fgsm_acc, fgsm_loss = gen_adv_examples(ensemble_model,
                                                     adv_loader, mifgsm,
                                                     loss_fn)
print(f'mifgsm_acc = {fgsm_acc:.5f}, mifgsm_loss = {fgsm_loss:.5f}')

create_dir(root, 'mifgsm', adv_examples, adv_names)

# os.system("")
os.system("cd mifgsm && tar zcvf ../mifgsm.tgz * > /dev/null")