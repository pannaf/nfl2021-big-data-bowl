import os
import util
import numpy as np
import random
import pandas as pd
import ipdb

random.seed(42)

def parse_imname_to_game_play_id(imname):
    game_play = imname.split('.')[0].split('-')
    game_id = int(game_play[0])
    play_id = int(game_play[1])
    return game_id, play_id

#_, _, imnames = os.walk('/data2/Code/nfl_analytics/2021/pictorial_trajectories_v3').__next__()
_,_, imnames = os.walk('/data2/Code/nfl_analytics/2021/trajpict/pictTraj_centered_from_snap/pass_arrived').__next__()

gameIds = []
playIds = []

for imname in imnames:
    gameId, playId = parse_imname_to_game_play_id(imname)
    gameIds.append(gameId)
    playIds.append(playId)

df = pd.DataFrame.from_dict({'gameId': gameIds, 'playId': playIds})

games_unique = df['gameId'].unique()

num_train = 166
num_dev = 24
num_test = 47

df_train = util.load_csv('/data2/Code/nfl_analytics/2021/data/data_splits', 'train_split_pass_arrived.csv') 
df_dev = util.load_csv('/data2/Code/nfl_analytics/2021/data/data_splits', 'dev_split_pass_arrived.csv') 
df_test = util.load_csv('/data2/Code/nfl_analytics/2021/data/data_splits', 'test_split_pass_arrived.csv') 

test_games = df_test['gameId'].unique().tolist() + [2018123012, 2018123001, 2018123000, 2018123011, 2018123008, 2018123014, 2018123009, 2018123015, 2018123006, 2018123004,2018123003, 2018123002, 2018123007, 2018123013, 2018123010,2018123005]

df_test_revised = df[df['gameId'].isin(test_games)]

#df_train.to_csv('train_split_pass_arrived_v2.csv', index=False)
#df_dev.to_csv('dev_split_pass_arrived_v2.csv', index=False)
df_test_revised.to_csv('/data2/Code/nfl_analytics/2021/data/data_splits/test_split_pass_arrived_with_week17.csv', index=False)

print('Num train plays {}'.format(len(df_train)))
print('Num dev plays {}'.format(len(df_dev)))
print('Num test plays {}'.format(len(df_test_revised)))

ipdb.set_trace()

print('done?')
