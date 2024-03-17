import os
import subprocess

import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
DEVICE


torch.manual_seed(128)

Gr = torch.Generator()
Gr.manual_seed(256)

Gf = torch.Generator()
Gf.manual_seed(512)

Gv = torch.Generator()
Gv.manual_seed(1024)

### Mock setting ###
import logging
import requests
import tqdm
from torch.utils.data import Subset
from torchvision import transforms

USE_MOCK: bool = False

if USE_MOCK:
    logging.warning('Running with Mock')
    logging.warning('In this mode, internet access may be required.')

    n_checkpoints = 5
    
    mock_dir = './mock'
    mock_model_path = os.path.join(mock_dir, "weights_resnet18_cifar10.pth")
    os.makedirs(mock_dir, exist_ok=True)
### DEVICE check ###
if DEVICE != 'cuda':
    raise RuntimeError('Make sure you have added an accelerator to your notebook; the submission will fail otherwise!')


if USE_MOCK:
    
    class DatasetWrapper(Dataset):
        
        def __init__(self, ds: Dataset):
            self._ds = ds
    
        def __len__(self):
            return len(self._ds)
    
        def __getitem__(self, index):
            item = self._ds[index]
            result = {
                'image': item[0],
                'image_id': index,
                'age_group': item[1],
                'age': item[1],
                'person_id': index,
            }
            return result
    
    def get_dataset(batch_size, retain_ratio=0.8, thinning_param: int=1000, root=mock_dir) -> tuple[DataLoader, DataLoader, DataLoader]:
        
        # utils
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # create dataset
        train_ds = DatasetWrapper(torchvision.datasets.CIFAR10(root=mock_dir, train=True, download=True, transform=normalize))
        retain_ds = Subset(train_ds, range(0, int(len(train_ds)*retain_ratio), thinning_param))
        forget_ds = Subset(train_ds, range(int(len(train_ds)*retain_ratio), len(train_ds), thinning_param))
        val_ds = DatasetWrapper(torchvision.datasets.CIFAR10(root=mock_dir, train=False, download=True, transform=normalize))

        retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)
        forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

        return retain_loader, forget_loader, validation_loader
    

else:
    def load_example(df_row):
        image = torchvision.io.read_image(df_row['image_path'])
        result = {
            'image': image,
            'image_id': df_row['image_id'],
            'age_group': df_row['age_group'],
            'age': df_row['age'],
            'person_id': df_row['person_id']
        }
        return result


    class HiddenDataset(Dataset):
        '''The hidden dataset.'''
        def __init__(self, split='train'):
            super().__init__()
            self.examples = []

            df = pd.read_csv(f'/input/neurips-2023-machine-unlearning/{split}.csv')
            df['image_path'] = df['image_id'].apply(lambda x: os.path.join('/input/neurips-2023-machine-unlearning/', 'images', x.split('-')[0], x.split('-')[1] + '.png'))
            df = df.sort_values(by='image_path')
            df.apply(lambda row: self.examples.append(load_example(row)), axis=1)
            if len(self.examples) == 0:
                raise ValueError('No examples.')

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            example = self.examples[idx]
            image = example['image']
            image = image.to(torch.float32)
            example['image'] = image
            return example


    def get_dataset(batch_size):
        '''Get the dataset.'''
        retain_ds = HiddenDataset(split='retain')
        forget_ds = HiddenDataset(split='forget')
        val_ds = HiddenDataset(split='validation')

        retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True, generator=Gr)
        forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True, generator=Gf)
        validation_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, generator=Gv)

        return retain_loader, forget_loader, validation_loader


from contextlib import contextmanager
import time

@contextmanager
def stopwatch(name='STOPWATCH'):
    s = time.time()
    try:
        yield
    finally:
        print(f"{name}: {time.time()-s} seconds passed")

def knowledge_loss(student_outputs, labels, teacher_outputs):
    alpha=0.1
    T=0.5
    
    student_loss=F.cross_entropy(input=student_outputs, target=labels)
    distillation_loss=nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T)
    tot_loss=alpha*student_loss + (1-alpha)*distillation_loss
    
    return tot_loss

def tch_net(retain_loader, forget_loader):
    mynet=resnet18(weights=None, num_classes=10).to(DEVICE)
    
    epoch=1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mynet.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    
    mynet.train()
    for i in range(epoch):
        mynet.train()
        for retain in retain_loader: # train_loader 만들기
            for forget in forget_loader:
                inputs=torch.cat([retain['image'],forget['image']])
                targets=torch.cat([retain['age_group'],forget['age_group']])
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = mynet(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
        scheduler.step()
        
    mynet.eval()
    
    return mynet

def unlearning(
    net, 
    retain_loader, 
    forget_loader, 
    val_loader):
    """Simple unlearning by finetuning."""
    epochs = 1
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    full_model=tch_net(retain_loader,forget_loader)
    
    net.train()
    
    for ep in range(epochs):
        net.train()
        for sample in retain_loader:
            inputs = sample["image"]
            targets = sample["age_group"]
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
            optimizer.zero_grad()
            outputs = net(inputs)
            #loss = criterion(outputs, targets)
            tch_outputs=full_model(inputs)
            loss=knowledge_loss(outputs, targets, tch_outputs)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    net.eval()
    
    return net

if USE_MOCK:
    
    # NOTE: Almost same as the original codes
    
    # Download
    if not os.path.exists(mock_model_path):
        response = requests.get("https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth")
        open(mock_model_path, "wb").write(response.content)    
    
    os.makedirs('/tmp', exist_ok=True)
    retain_loader, forget_loader, validation_loader = get_dataset(64)
    net = resnet18(weights=None, num_classes=10)
    net.to(DEVICE)
    for i in tqdm.trange(n_checkpoints):
        net.load_state_dict(torch.load(mock_model_path))
        net_ = unlearning(net, retain_loader, forget_loader, validation_loader)
        state = net_.state_dict()
        torch.save(state, f'/tmp/unlearned_checkpoint_{i}.pth')

    # Ensure that submission.zip will contain exactly 512 checkpoints 
    # (if this is not the case, an exception will be thrown).
    unlearned_ckpts = os.listdir('/tmp')
    if len(unlearned_ckpts) != n_checkpoints:
        raise RuntimeError('Expected exactly 512 checkpoints. The submission will throw an exception otherwise.')

    subprocess.run('zip submission.zip /tmp/*.pth', shell=True)
    
else:
    if os.path.exists('/input/neurips-2023-machine-unlearning/empty.txt'):
        # mock submission
        subprocess.run('touch submission.zip', shell=True)
    else:

        # Note: it's really important to create the unlearned checkpoints outside of the working directory 
        # as otherwise this notebook may fail due to running out of disk space.
        # The below code saves them in /tmp to avoid that issue.

        os.makedirs('/tmp', exist_ok=True)
        retain_loader, forget_loader, validation_loader = get_dataset(64)
        net = resnet18(weights=None, num_classes=10)
        net.to(DEVICE)
        for i in range(512):
            net.load_state_dict(torch.load('/input/neurips-2023-machine-unlearning/original_model.pth'))
            net_ = unlearning(net, retain_loader, forget_loader, validation_loader)
            state = net_.state_dict()
            torch.save(state, f'/tmp/unlearned_checkpoint_{i}.pth')

        # Ensure that submission.zip will contain exactly 512 checkpoints 
        # (if this is not the case, an exception will be thrown).
        unlearned_ckpts = os.listdir('/tmp')
        if len(unlearned_ckpts) != 512:
            raise RuntimeError('Expected exactly 512 checkpoints. The submission will throw an exception otherwise.')

        subprocess.run('zip submission.zip /tmp/*.pth', shell=True)


