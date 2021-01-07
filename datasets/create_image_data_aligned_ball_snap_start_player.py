import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
import numpy as np
import scipy.misc
import imageio
import os

from joblib import Parallel, delayed
import multiprocessing

import util
import ipdb

# rootdir = '../data'
# df_plays = util.load_csv(rootdir, 'plays.csv')
# df_games = util.load_csv(rootdir, 'games.csv')
# df_targeted_receiver = util.load_csv(rootdir, 'targetedReceiver.csv')

def generate_image(df_defense, df_offense, df_football):
    padding_x = 100
    padding_y = 70
    img = np.zeros((160+3*padding_y, 120*3+3*padding_x, 3))

    for player in df_defense['displayName'].unique():
        df_ = df_defense[df_defense['displayName'].isin([player])].sort_values('frameId')
        xy_player = np.array(df_[['x_centered','y_centered']])
        for i, xy_ in enumerate(xy_player):
            img[int(xy_[1]*3), int(xy_[0]*3), 0] = 255*(i+0.)/xy_player.shape[0]

    for player in df_offense['displayName'].unique():
        df_ = df_offense[df_offense['displayName'].isin([player])].sort_values('frameId')
        xy_player = np.array(df_[['x_centered','y_centered']])
        for i, xy_ in enumerate(xy_player):
            img[int(xy_[1]*3), int(xy_[0]*3), 1] = 255*(i+0.)/xy_player.shape[0]

    xy_football = np.array(df_football[['x_centered','y_centered']])
    for i, xy_ in enumerate(xy_football):
        img[int(xy_[1]*3), int(xy_[0]*3), 2] = 255*(i+0.)/xy_football.shape[0]

    sigma = 3
    img_blurred = img
    for i in range(3):
        img_blurred[:,:,i] = gaussian_filter(img[:,:,i], sigma=sigma)
    img_blurred[:,:,2] = img_blurred[:,:,2] * 2

    #plt.imshow(img_blurred[::-1,:,:]/np.max(img_blurred)) 
    temp_im = 255*img_blurred[::-1,:,:]/np.max(img_blurred)
    temp_im = temp_im.astype('uint8')
    return temp_im


def save_week_images(week):
#for week in [1]: #range(1, 18):
    event_type = 'pass_arrived'
    save_dir = '/data2/Code/nfl_analytics/2021/trajpict_PLAYER/pictTraj_centered_from_snap/{}'.format(event_type)
    rootdir = '../data'
    df_plays = util.load_csv(rootdir, 'plays.csv')
    df_games = util.load_csv(rootdir, 'games.csv')
    df_targeted_receiver = util.load_csv(rootdir, 'targetedReceiver.csv')

    print('Week {}'.format(week))
    df_tracks = util.load_csv(rootdir, 'week{}.csv'.format(week)) 
    
    df_game_play = df_tracks[['gameId','playId']].groupby(['gameId','playId']) \
                                                 .size() \
                                                 .reset_index()[['gameId', 'playId']]
    
    for idx, row in df_game_play.iterrows(): 
        gameId = row['gameId']
        playId = row['playId']
        print('Week {}, Game id {}, play id {}'.format(week, gameId, playId))

        if os.path.exists('{}/{}-{:04d}.png'.format(save_dir, gameId, playId)):
            continue
        
        df_play = util.get_slice_by_id(df_tracks, playId, gameId)
        df_play = util.center_play(df_play)
        df_play['x_centered'] += 120
        df_play['y_centered'] += 62 
        assert min(df_play['y_centered']) >= 0, 'Uh oh {}, {}'.format(gameId, playId)
        # df_play['x_centered'] = df_play['x']
        # df_play['y_centered'] = df_play['y']

        ## just save trajectories up to moment of pass arrived
        ## for incomplete passes that don't have pass_arrived flag, then
        ## grab frame just before event is pass_outcome_incomplete
        idx_snap = df_play[df_play['event'].isin(['ball_snap'])]['frameId'].unique()[0]
        if event_type == 'play_end':
            idx_arrived = max(df_play['frameId'].unique())
        if event_type == 'ball_snap':
            idx_arrived = df_play[df_play['event'].isin([event_type])]['frameId'].unique()[0]
            idx_snap = idx_snap-1
        if event_type is not 'ball_snap' and event_type is not 'play_end':
            idx_arrived = df_play[df_play['event'].isin([event_type])]['frameId'].unique()
            if len(idx_arrived) == 1:
                idx_arrived = idx_arrived[0]
            else:
                if event_type == 'pass_arrived':
                    incomplete_flag = util.get_slice_by_id(df_plays, playId, gameId)['passResult'].values[0]=='I'
                    if len(idx_arrived) == 0 and incomplete_flag:
                        idx_arrived = df_play[df_play['event'].isin(['pass_outcome_incomplete'])]['frameId'].unique()
                        if len(idx_arrived) ==1:
                            idx_arrived = idx_arrived[0]-1
                        else:
                            continue
                    else:
                        continue
                else: 
                    continue
            # include the frame where the pass arrived event occurs
        df_play = df_play[df_play['frameId'].isin(range(idx_snap, idx_arrived + 1))]
        
        #df_play_centered = util.center_play(self.play_df)
       
        df_game = df_games[df_games['gameId'].isin([gameId])]
        possession_team = util.get_slice_by_id(df_plays, playId, gameId)['possessionTeam'].values[0]
        offense, defense = util.get_offense_defense(df_game, possession_team)
        
        df_offense, _ = util.get_coords_by_team(df_play, offense)
        df_defense, _ = util.get_coords_by_team(df_play, defense)
        df_football, _ = util.get_coords_by_team(df_play, 'football')
       
        for nflId in df_defense['nflId'].unique():
            nflIds = [idx for idx in df_defense['nflId'].unique() if idx != nflId] 
            df_defense_subset = df_defense[df_defense['nflId'].isin(nflIds)]
            Im = generate_image(df_defense_subset, df_offense, df_football)


            # save_dir = '/data2/Code/nfl_analytics/2021/pictTraj_centered_from_snap/{}'.format(event_type)        
            imageio.imwrite('{}/{}-{:04d}_{}.png'.format(save_dir, gameId, playId, int(nflId)), Im)

        #ipdb.set_trace()

Parallel(n_jobs = multiprocessing.cpu_count()-1)(delayed(save_week_images)(ii) for ii in range(1,18)) 
