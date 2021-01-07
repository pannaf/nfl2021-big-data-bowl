import pandas as pd
import util
import os
import ipdb

df_byplay_split = util.load_csv('/data2/Code/nfl_analytics/2021/data/data_splits/', 'test_split_pass_arrived_with_week17.csv')

_, _, imnames = os.walk('/data2/Code/nfl_analytics/2021/trajpict_PLAYER/pictTraj_centered_from_snap/pass_arrived').__next__()

def parse_imname_to_game_play_id(imname):
    game_play = imname.split('.')[0].split('-')
    game_id = int(game_play[0])
    play_id = int(game_play[1].split('_')[0])
    nfl_id = int(game_play[1].split('_')[1])
    return game_id, play_id, nfl_id


gameIds = []
playIds = []
nflIds = []

for imname in imnames:
    gameId, playId, nflId = parse_imname_to_game_play_id(imname)
    gameIds.append(gameId)
    playIds.append(playId)
    nflIds.append(nflId)

df_game_play_player = pd.DataFrame.from_dict({'gameId': gameIds,
                                              'playId': playIds,
                                              'nflId': nflIds})

df = pd.merge(df_game_play_player, df_byplay_split, on=['gameId', 'playId'])

df.to_csv('/data2/Code/nfl_analytics/2021/data/data_splits/test_split_pass_arrived_PLAYER_full.csv',index=False)

ipdb.set_trace()

print('done?')
