import torch
import torch.nn as nn
from torch.utils import data
from motion_models.dataloader import PlayDataset
import numpy as np
from skimage import io, transform
import ipdb
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import time
start = time.time()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

# Parameters for data loader
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 2 
          }
params_dev = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 2 
          }
# Hyper-parameters
num_epochs = 30 #10 #10
input_size = 75
hidden_size = 25
num_classes = 13
learning_rate = 0.001 # alexnet used 0.001

# Datasets
# im_rootdir = '/data2/Code/nfl_analytics/2021/pictorial_trajectories_v5'

def evaluate(dataloader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        predicted_list = []
        groundtruth_list = []
        for (local_batch,local_labels) in dataloader:
            # Transfer to GPU
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)

            outputs = model(local_ims)
            _, predicted = torch.max(outputs.data, 1)
            total += local_labels.size(0)
            predicted_list.extend(predicted)
            groundtruth_list.extend(local_labels)
            correct += (predicted == local_labels).sum().item()
            #ipdb.set_trace()

        print('Accuracy of the network on the {} evaluation images: {} %'.format(total, 100 * correct / total))
    pl = [p.cpu().numpy().tolist() for p in predicted_list]
    gt = [p.cpu().numpy().tolist() for p in groundtruth_list]

    from sklearn.metrics import classification_report
    print(classification_report(gt, pl, digits=3))

    label_map = ['C', 'I'] #, 'S', 'IN', 'R']
    for id in range(len(label_map)):
        print('{}: {}'.format(label_map[id],sum([p and g for (p,g) in zip(np.array(pl)==np.array(gt),np.array(gt)==id)])/(sum(np.array(gt)==id)+0.)))


event_type = 'pass_arrived'

im_rootdir = '/data2/Code/nfl_analytics/2021/trajpict/pictTraj_centered_from_snap/{}'.format(event_type)
data_rootdir = '/data2/Code/nfl_analytics/2021/data'


## ball snap stats:
if event_type is 'ball_snap':
    print('Ball Snap')
    mean = [0.22185783632826273, 0.17329253537644185, 0.05810781520547683]
    std = [3.8430682508407914, 3.525189335135875, 2.733388403720904]

## pass forward
if event_type is 'pass_forward':
    print('Pass Forward')
    mean = [0.5154701721714169, 0.42373869342749004, 0.11667604609513321]
    std = [5.975045342280524, 5.114765323946453, 3.689086707138345]

## pass arrived
if event_type is 'pass_arrived':
    mean = [0.8278661085744993, 0.6397270091149045, 0]# 0.21311753731973818]
    std = [8.311023923412382, 6.806565451355073, 1] # 4.151762584107889]

if event_type is 'play_end':
    mean = [1.038661048777042, 0.747664668936503, 0.27118879995517553]
    std = [8.973929529255928, 7.132237691495664, 4.781272060488431]

    

# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
# Generators
im_size = 224 # i think alexnet needs to be 227
train_dataset = PlayDataset('train',im_rootdir, data_rootdir,
                                    transform=transforms.Compose([
                                               transforms.RandomVerticalFlip(),
                                               transforms.Resize((im_size, im_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)
                                           ]))
train_loader = data.DataLoader(train_dataset, **params)

dev_dataset = PlayDataset('dev',im_rootdir, data_rootdir,
                                transform=transforms.Compose([
                                           transforms.Resize((im_size, im_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)
                                       ]))

dev_loader = data.DataLoader(dev_dataset, **params_dev)

#model = NeuralNet().to(device)
#model = models.vgg16() #.to(device)
# NOTE: Alexnet
###model_conv = models.alexnet() #pretrained='imagenet')
#model_conv = models.vgg16()
# Number of filters in the bottleneck layer
alexnet = False
if alexnet:
    model_conv = models.alexnet() #pretrained='imagenet')
    num_ftrs = model_conv.classifier[6].in_features
    # convert all the layers to list and remove the last one
    features = list(model_conv.classifier.children())[:-1]
    ## Add the last layer based on the num of classes in our dataset
    n_class = 3 
    features.extend([nn.Linear(num_ftrs, n_class)])
    ## convert it into container and add it to our model class.
    model_conv.classifier = nn.Sequential(*features)
else:
    print('Using ResNet-18 model')
    model_conv = models.resnet18().train()
    # change number of input channels to 4
    ## model_conv.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
    # change number of output units to 2
    num_ftrs = model_conv.fc.in_features
    n_class = 2 
    model_conv.fc = nn.Linear(num_ftrs, n_class)

model = model_conv.to(device)

# Loss and optimizer
## NOTE: if you have imbalanced data, you can use class weights on the loss
class_weights = torch.FloatTensor(train_dataset.class_weights).to(device) # ,8.56]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# Train the model
# Loop over epochs
evaluate(dev_loader, model) 
total_step = len(train_loader)
print('Event {}'.format(event_type))
for epoch in range(num_epochs):
    # Training
    print('epoch {}'.format(epoch))
    for i, (local_batch, local_labels) in enumerate(train_loader):
        # Transfer to GPU
        local_ims, local_labels = local_batch.to(device), local_labels.to(device)
        #ipdb.set_trace()
        # Forward pass
        outputs = model(local_ims)
        loss = criterion(outputs, local_labels)

        # Backward and optimize
        # important to zero the gradient, or pytorch will keep accumulating gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    evaluate(dev_loader, model) 
    torch.save(model.state_dict(), 'pass_ckpts/{}/resnet18_224-{}_from_snap_no_fb.ckpt'.format(event_type,epoch))

end = time.time()
print('Time: {}'.format(end - start))


# Save the model checkpoint
#evaluate(dev_loader, model) 
#torch.save(model.state_dict(), 'passResult_checkpoints/{}/resnet18_224_10epochs_from_snap.ckpt'.format(event_type))
##torch.save(model.state_dict(), 'alexnet227_random_frames_20.ckpt')
