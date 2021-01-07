from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import ipdb
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform

gameId = 2018101500
playId = 2723
## incompletion example.
gameId = 2018122305
playId = 1135

gameId = 2018122305
playId = 769

## pass incompletion example
#gameId = 2018111110
#playId = 969

### this is a good pass completion example
#gameId = 2018102110
#playId = 2025
#
#gameId = 2018111102
#playId = 58
#
#gameId = 2018120905
#playId = 4090

## pass completion example
gameId = 2018123009
playId = 1271 

gameId = 2018091607
playId = 125

## example of an open receiver who misses the ball I,2018092300,1221. another example is 2018091609-3123


impath = '/data2/Code/nfl_analytics/2021/trajpict/pictTraj_centered_from_snap/pass_arrived/{}-{:04d}.png'.format(gameId, playId)
image = Image.open(impath)
imshow(image)
plt.savefig('{}-{:04d}_pass_arrived.png'.format(gameId, playId))

# Imagenet mean/std

normalize = transforms.Normalize(
   mean = [0.8278661085744993, 0.6397270091149045, 0.21311753731973818],
   std = [8.311023923412382, 6.806565451355073, 4.151762584107889]

)

# Preprocessing - scale to 224x224 for model, convert to tensor, 
# and normalize to -1..1 with mean/std for ImageNet

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

display_transform = transforms.Compose([
   transforms.Resize((370,660))])
display_transform_tensor = transforms.Compose([
   transforms.Resize((370,660)),
   transforms.ToTensor()])

tensor = preprocess(image)

prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)


model = models.resnet18().eval()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('/data2/Code/nfl_analytics/2021/nfl2021/pass_outcome_models/pass_arrived.ckpt'))

model.cuda()
#model.eval()

class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = ((output.cpu()).data).numpy()
    def remove(self): 
        self.hook.remove()

final_layer = model._modules.get('layer4')

activated_features = SaveFeatures(final_layer)

prediction = model(prediction_var)
pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove()

topk(pred_probabilities,1)

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

weight_softmax_params = list(model._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

class_idx = topk(pred_probabilities,1)[1].int()
class_labels = ['C', 'I']
print('Class IDX: {}, which is {}'.format(class_idx, class_labels[class_idx]))

overlay = getCAM(activated_features.features, weight_softmax, class_idx )


imshow(display_transform(image))
imshow(skimage.transform.resize(overlay[0], display_transform_tensor(image).shape[1:3]), alpha=0.5, cmap='jet');
plt.axis('off')
plt.savefig('{}-{:04d}_pass_arrived_cam.png'.format(gameId, playId))
#plt.show()

#ipdb.set_trace()

print('done')


