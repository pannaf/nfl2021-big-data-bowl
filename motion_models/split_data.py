import os
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

_, _, imnames = os.walk('/data2/Code/nfl_analytics/2021/pictorial_trajectories_v3').__next__()

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

# in place shuflfe the games
random.shuffle(games_unique)

train_games = games_unique[:num_train]
dev_games = games_unique[num_train:num_train+num_dev]
test_games = games_unique[num_train+num_dev:]

df_train = df[df['gameId'].isin(train_games)]
df_dev = df[df['gameId'].isin(dev_games)]
df_test = df[df['gameId'].isin(test_games)]

df_train.to_csv('train_split_pass_arrived_v2.csv', index=False)
df_dev.to_csv('dev_split_pass_arrived_v2.csv', index=False)
df_test.to_csv('test_split_pass_arrived_v2.csv', index=False)

print('Num train plays {}'.format(len(df_train)))
print('Num dev plays {}'.format(len(df_dev)))
print('Num test plays {}'.format(len(df_test)))
