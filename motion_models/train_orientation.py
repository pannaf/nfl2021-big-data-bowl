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
# Hyper-parameters
num_epochs = 10 #10 #10
input_size = 75
hidden_size = 25
num_classes = 13
learning_rate = 0.001 # alexnet used 0.001

# Datasets
im_rootdir = '/data2/Code/nfl_analytics/2021/pictorial_trajectories_v5'
data_rootdir = '/data2/Code/nfl_analytics/2021/data'

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
# Generators
im_size = 224 # i think alexnet needs to be 227
train_dataset = PlayDataset('train',im_rootdir, data_rootdir,
                                    transform=transforms.Compose([
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

dev_loader = data.DataLoader(dev_dataset, **params)

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
    print('Using ResNet-50 model')
    model_conv = models.resnet50()
    num_ftrs = model_conv.fc.in_features
    n_class = 2 
    model_conv.fc = nn.Linear(num_ftrs, n_class)


model = model_conv.to(device)

# Loss and optimizer
## NOTE: if you have imbalanced data, you can use class weights on the loss
class_weights = torch.FloatTensor([1.85,1]).to(device) # ,8.56]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# Loop over epochs
total_step = len(train_loader)
for epoch in range(num_epochs):
    # Training
    print('epoch {}'.format(epoch))
    for i, (local_batch, local_labels) in enumerate(train_loader):
        print(i)
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

        if (i+1) % 4 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

end = time.time()
print('Time: {}'.format(end - start))

# NOTE: the evaluation should really live in a separate evaluation file
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    predicted_list = []
    groundtruth_list = []
    for (local_batch,local_labels) in dev_loader:
        # Transfer to GPU
        local_ims, local_labels = local_batch.to(device), local_labels.to(device)

        outputs = model(local_ims)
        _, predicted = torch.max(outputs.data, 1)
        total += local_labels.size(0)
        predicted_list.extend(predicted)
        groundtruth_list.extend(local_labels)
        correct += (predicted == local_labels).sum().item()
        #ipdb.set_trace()

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
pl = [p.cpu().numpy().tolist() for p in predicted_list]
gt = [p.cpu().numpy().tolist() for p in groundtruth_list]

label_map = ['I', 'C', 'S', 'IN', 'R']
for id in range(len(label_map)):
    print('{}: {}'.format(label_map[id],sum([p and g for (p,g) in zip(np.array(pl)==np.array(gt),np.array(gt)==id)])/(sum(np.array(gt)==id)+0.)))


# Save the model checkpoint
torch.save(model.state_dict(), 'resnet50_224_10epochs_ball_snaps.ckpt')
#torch.save(model.state_dict(), 'alexnet227_random_frames_20.ckpt')
