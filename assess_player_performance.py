import pandas as pd
import numpy as np
import ipdb

df_player_list = pd.read_csv('/data2/Code/nfl_analytics/2021/data/players.csv')

df_player = pd.read_csv('/data2/Code/nfl_analytics/2021/nfl2021/pass_arrived_pass_result_prob_PLAYER_full.csv')

df_team = pd.read_csv('/data2/Code/nfl_analytics/2021/nfl2021/pass_outcome_predictions/pass_arrived.csv')

df = pd.merge(df_player, df_team, how='left', on=['gameId', 'playId', 'passResult'])
df = pd.merge(df, df_player_list, how='left', on=['nflId'])

df_game_play = df[['gameId','playId']].groupby(['gameId','playId']) \
                                      .size() \
                                      .reset_index()[['gameId', 'playId']]

df['p_Complete delta'] = df['p_Complete pass_arrived player'] - df['p_Complete pass_arrived']
df['p_Complete delta v league average'] = df['p_Complete delta'] - np.mean(df['p_Complete delta'])

#players_to_include = df['displayName'].value_counts().index[df['displayName'].value_counts().values>=50].values.tolist()
#
##ipdb.set_trace()
#df = df[df['displayName'].isin(players_to_include)]


df = df[df['passResult'].isin(['I'])]

players_to_include = df['displayName'].value_counts().index[df['displayName'].value_counts().values>30].values.tolist()
#ipdb.set_trace()
df = df[df['displayName'].isin(players_to_include)]

df_delta_by_player = df[['nflId','p_Complete delta','p_Complete delta v league average','displayName', 'position']].groupby(['nflId','displayName', 'position']).describe() 

df_stats = df_delta_by_player['p_Complete delta'][['mean','count', 'std']]
df_stats_v2 = df_delta_by_player['p_Complete delta v league average'][['mean','count']]
#df_stats = df_stats[df_stats['count']>=50]

pd.options.display.max_rows = 999
df_sorted = df_stats.sort_values('mean',ascending=False)
df_sorted_v2 = df_stats_v2.sort_values('mean',ascending=False)
#.sort_values(by='mean',ascending=False)

#df_player_stats = pd.merge(df_delta_by_player, df_player_list[['nflId','displayName']], on=['nflId'])

ipdb.set_trace()


print('done?')

