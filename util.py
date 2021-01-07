import matplotlib
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import ipdb


def load_csv(rootdir, filename):
    df = pd.read_csv('{}/{}'.format(rootdir, filename))
    return df

def get_slice_by_id(df, play_id, game_id):
    temp = df[df['playId'].isin([play_id])]
    return temp[temp['gameId'].isin([game_id])]

def center_play(df_tracks):
    football_0 = df_tracks[df_tracks['frameId'].isin([1])]
    football_0 = football_0[football_0['displayName'].isin(['Football'])]
    x0 = football_0['x'].values[0]
    y0 = football_0['y'].values[0]
    x_centered = df_tracks['x'] - x0
    y_centered = df_tracks['y'] - y0

    scale = 2*(df_tracks['playDirection']=='left')-1

    df_tracks['x_centered'] = scale * x_centered
    df_tracks['y_centered'] = scale * y_centered

    return df_tracks

def get_coords_by_team(df, team):
    df_team = df[df['team'].isin([team])]
    if team is not 'football':
        num_players = df_team['displayName'].unique().shape[0]
    else:
        num_players = 1
    return df_team[['x', 'y', 'x_centered','y_centered','o','frameId','position','displayName', 'nflId']], num_players


def get_offense_defense(df_game, possessionTeam):
    if possessionTeam == df_game['homeTeamAbbr'].values[0]:
        offense = 'home'
        defense = 'away'
    else:
        offense = 'away'
        defense = 'home'
    return offense, defense


def draw_field(line_width, title):
    field_color = 'dimgray'
    yard_lines = ['','','10','20','30','40','50','40','30','20','10']
    ls = [0,10,20,30,40,50,60,70,80,90,100,110,120]
    ls = [l for l in ls]
    ws = [0,160/3.]
    ws = [w for w in ws]
    #hs = [70.75,89.25]
    
    ## 0 -> 120 yards on the x-axis
    ## 0 -> 53.33 yards on the y-axis
    
    fig = plt.figure(figsize=(120/5, 53.33/5))
    ax = fig.add_subplot(111)
    
    # endzones
    rect_left_endzone = matplotlib.patches.Rectangle((0,0), 10, 53.33, color=field_color)
    rect_right_endzone = matplotlib.patches.Rectangle((110,0), 10, 53.33, color=field_color)
    rect_field = matplotlib.patches.Rectangle((10,0), 100, 53.33, color='black')
    
    ax.add_patch(rect_left_endzone)
    ax.add_patch(rect_right_endzone)
    ax.add_patch(rect_field)
    for (x,yardline) in zip(ls,yard_lines):
        if x == 0 or x == 330:
            color_ = 'darkgray'
        else:
            color_ = 'dimgray'
        plt.plot([x, x], [ws[0], ws[1]], linewidth=line_width, color=color_)
        plt.text(x-1.6, 53.33/2.-1, yardline, fontsize=24,color='dimgray')
    for y in ws:
        plt.plot([ls[0], ls[-1]], [y, y], linewidth=line_width, color='darkgray')
    xhashes = []
    y1hashes = []
    y2hashes = []
    for hx in range(50):
        if ((hx/50*300/3.)+30/3.)%10 != 0:
            xhashes.append((300/3.*hx/50)+10)
            y1hashes.append(70.75/3.)
            y2hashes.append(89.25/3.)
    plt.plot([xhashes], [y1hashes], color=field_color, marker='|',)
    plt.plot([xhashes], [y2hashes], color=field_color, marker='|',)
    plt.axis('off')
    plt.title(title)
    #plt.show()
    return fig, ax
