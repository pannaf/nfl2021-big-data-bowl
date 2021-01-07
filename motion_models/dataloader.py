from __future__ import print_function, division
import json
import ipdb
import os
import torch
import time
import math
import random
import util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import imread
from skimage import io
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# ipdb.set_trace()


class PlayDataset(Dataset):
    def __init__(self, split, im_rootdir, data_rootdir, transform=None):
        self.transform = transform
        # using this for getting the pass outcome
        df_coverages = util.load_csv(data_rootdir, 'coverages_week1.csv')
        df_plays = util.load_csv(data_rootdir, 'plays.csv')
        df_plays_subset = df_plays[['gameId', 'playId', 'passResult']]
        if split != 'test':
            df_impaths = util.load_csv(data_rootdir, '/data_splits/{}_split_pass_arrived.csv'.format(split))
        else:
            df_impaths = util.load_csv(data_rootdir, '/data_splits/{}_split_pass_arrived_with_week17.csv'.format(split))
        self.im_rootdir = im_rootdir
        self.data = pd.merge(df_impaths, df_plays_subset, how="left", on=["gameId","playId"])        
        # drop anything with NaN passResult
        self.data = self.data.dropna()
        self.data = self.data[self.data['passResult'].isin(['I', 'C'])] #, 'S'])]

        self.label_column_name = 'passResult'
        self.class_labels = self.data[self.label_column_name].value_counts().index.tolist()
        self.class_weights = 1/self.data[self.label_column_name].value_counts(normalize=True).values
        self.class_weights = self.class_weights/min(self.class_weights).tolist()

        _, _, imnames = os.walk(im_rootdir).__next__()
        
        gameIds = []
        playIds = []
        
        for imname in imnames:
            gameId, playId = self.__parse_imname_to_game_play_id__(imname)
            gameIds.append(gameId)
            playIds.append(playId)
        
        df = pd.DataFrame.from_dict({'gameId': gameIds, 'playId': playIds})

        self.data = pd.merge(self.data, df, how ='inner', on =['gameId', 'playId'])

        self.passResult_dict = {'C': 0,
                                'I': 1
                                }  
        # TODO: we can restrict the data here to only contain 'I', 'C', 'IN' passResults

        print('loaded image_paths and pass outcome')

    def __parse_imname_to_game_play_id__(self, imname):
        game_play = imname.split('.')[0].split('-')
        game_id = int(game_play[0])
        play_id = int(game_play[1])
        return game_id, play_id
        

    def __len__(self):
        return len(self.data)     

    def __getitem__(self, idx):
        data_slice = self.data.iloc[idx]
        # Get image
        image_path = '{}/{}-{:04d}.png'.format(self.im_rootdir, data_slice['gameId'], data_slice['playId'])
        image = io.imread(image_path)
        ### zero out the football for the pass arrived moment
        image[:,:,-1] = 0
        if self.transform:
            PIL_image = Image.fromarray(image)
            image = self.transform(PIL_image)
        # Get label
        label = self.passResult_dict[data_slice['passResult']]
        return image, label

def unit_dev_load_all_data(data_split=None):
    im_rootdir = '/data2/Code/nfl_analytics/2021/trajpict_orientation/pictTraj_centered_from_snap/pass_arrived'
    data_rootdir = '/data2/Code/nfl_analytics/2021/data'
    split = 'train' # could be 'dev' or could be 'test'
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    data = PlayDataset(split, im_rootdir, data_rootdir) #, transform=transforms.Compose([
                                      #         transforms.Resize((227, 227)),
                                      #         transforms.ToTensor(),
                                      #         transforms.Normalize(mean, std)
                                      #     ]))

    mean_ch1 = []
    mean_ch2 = []
    mean_ch3 = []
    mean_ch4 = []
    std_ch1 = []
    std_ch2 = []
    std_ch3 = []
    std_ch4 = []
    for i, (image, label) in enumerate(data):
        print(i)
        mean_ch1.append(np.mean(image[:,:,0]))
        mean_ch2.append(np.mean(image[:,:,1]))
        mean_ch3.append(np.mean(image[:,:,2]))
        mean_ch4.append(np.mean(image[:,:,3]))
        std_ch1.append(np.std(image[:,:,0]))
        std_ch2.append(np.std(image[:,:,1]))
        std_ch3.append(np.std(image[:,:,2]))
        std_ch4.append(np.std(image[:,:,3]))
    print('mean = [{}, {}, {}, {}]'.format(np.mean(mean_ch1), np.mean(mean_ch2), np.mean(mean_ch3), np.mean(mean_ch4)))
    print('std = [{}, {}, {}, {}]'.format(np.mean(std_ch1), np.mean(std_ch2), np.mean(std_ch3), np.mean(std_ch4)))
    ipdb.set_trace()


    #im_rootdir = '/data2/Code/nfl_analytics/2021/trajpict/pictorial_trajectories_v4'
    #data_rootdir = '/data2/Code/nfl_analytics/2021/data'
    #split = 'train' # could be 'dev' or could be 'test'
    #mean = [0.5, 0.5, 0.5]
    #std = [0.5, 0.5, 0.5]
    #data = PlayDataset(split, im_rootdir, data_rootdir, transform=None) #transforms.Compose([
    #                                  #          transforms.Resize((227, 227)),
    #                                  #          transforms.ToTensor(),
    #                                  #          transforms.Normalize(mean, std)
    #                                  #      ]))

    #for i, (image, label) in enumerate(data):
    #    print(i)
    #    ipdb.set_trace()

def main():
    unit_dev_load_all_data()

if __name__ == "__main__":
    main()
