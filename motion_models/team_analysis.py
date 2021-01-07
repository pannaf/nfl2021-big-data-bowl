import torch
import pandas as pd
import torch.nn as nn
from torch.utils import data
from motion_models.dataloader_team_analysis import PlayDataset
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
params_eval = {'batch_size': 64,
               'shuffle': False,
               'num_workers': 2 
               }
# Hyper-parameters
num_epochs = 10 #10 #10
input_size = 75
hidden_size = 25
num_classes = 13
learning_rate = 0.001 # alexnet used 0.001

# Datasets
# im_rootdir = '/data2/Code/nfl_analytics/2021/pictorial_trajectories_v5'

def evaluate(dataloader, model, event_type):
    with torch.no_grad():
        softmax = nn.Softmax(dim=1)
        correct = 0
        total = 0
        predicted_list = []
        groundtruth_list = []

        passResult = []
        gameIds = []
        playIds = []
        probCom = []
        probInc = []
        label_map = ['C', 'I']

        for (local_batch,local_labels,local_games,local_plays) in dataloader:
            # Transfer to GPU
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)

            outputs = model(local_ims)
            probability = softmax(outputs)
            
            passResult.extend([label_map[pr] for pr in local_labels.cpu().numpy().tolist()])
            gameIds.extend(local_games.cpu().numpy().tolist())
            playIds.extend(local_plays.cpu().numpy().tolist())
            probCom.extend(probability[:,0].cpu().numpy().tolist())
            probInc.extend(probability[:,1].cpu().numpy().tolist())

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

    for id in range(len(label_map)):
        print('{}: {}'.format(label_map[id],sum([p and g for (p,g) in zip(np.array(pl)==np.array(gt),np.array(gt)==id)])/(sum(np.array(gt)==id)+0.)))

    results_df = {'passResult': passResult,
                  'gameId': gameIds,
                  'playId': playIds,
                  'p_Complete {}'.format(event_type): probCom,
                  'p_Incomplete {}'.format(event_type): probInc
                 }
    results_df = pd.DataFrame.from_dict(results_df)
    return results_df

event_epoch_dict = {'ball_snap': 3,
                    'pass_forward': 4,
                    'pass_arrived': 4, # other pass arrived is 2
                    'play_end': 7
                   }

event_type = 'ball_snap'
#event_type = 'pass_forward'
#event_type = 'pass_arrived'
#event_type = 'play_end'

im_rootdir = '/data2/Code/nfl_analytics/2021/trajpict/pictorial_traj_centered/{}'.format(event_type)
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
    mean = [0.8278661085744993, 0.6397270091149045, 0.21311753731973818]
    std = [8.311023923412382, 6.806565451355073, 4.151762584107889]

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

dev_loader = data.DataLoader(dev_dataset, **params_eval)
test_dataset = PlayDataset('test',im_rootdir, data_rootdir,
                                transform=transforms.Compose([
                                           transforms.Resize((im_size, im_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)
                                       ]))

test_loader = data.DataLoader(test_dataset, **params_eval)

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
    model_conv = models.resnet18().eval()
    num_ftrs = model_conv.fc.in_features
    n_class = 2 
    model_conv.fc = nn.Linear(num_ftrs, n_class)

model = model_conv.to(device)

#for epoch in range(1,10):

# ball_snap is epoch 2
# pass_forward is epoch 1
# pass_arrived is epoch 8
# play_end is epoch 9 

print('Event {}'.format(event_type))
ckpt_path = '/data2/Code/nfl_analytics/2021/nfl2021/pass_outcome_predictor_models/{}.ckpt'
model.load_state_dict(torch.load(ckpt_path.format(event_type)))

#for epoch in range(1,10):
#epoch = event_epoch_dict[event_type] 
#print('Epoch {}'.format(epoch))
#model.load_state_dict(torch.load('/data2/Code/nfl_analytics/2021/nfl2021/passResult_checkpoints/{}/resnet18_224-{}_from_snap.ckpt'.format(event_type, epoch)))
#
## Loss and optimizer
### NOTE: if you have imbalanced data, you can use class weights on the loss
#class_weights = torch.FloatTensor(train_dataset.class_weights).to(device) # ,8.56]).to(device)
#criterion = nn.CrossEntropyLoss(weight=class_weights)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# Loop over epochs
print('Development set evaluation')
evaluate(dev_loader, model, event_type) 
print('Test set evaluation')
results_df = evaluate(test_loader, model, event_type) 
results_df.to_csv('{}_pass_result_prob.csv'.format(event_type), index=False)

ipdb.set_trace()

print('done?')

