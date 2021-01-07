import util
import ipdb
import numpy as np
import pandas as pd

rootdir = '/data2/Code/nfl_analytics/2021/data'

df_games = util.load_csv(rootdir, 'games.csv')
df_plays = util.load_csv(rootdir, 'plays.csv')

df_plays_teams = pd.merge(df_plays, df_games, how="left", on=["gameId"])
df_plays_teams['defenseTeam'] = df_plays_teams.apply(lambda x: \
                                   x['homeTeamAbbr'] if x['visitorTeamAbbr'] \
                                   == x['possessionTeam'] else x['visitorTeamAbbr'], axis=1)

event_types = ['ball_snap', 'pass_forward', 'pass_arrived', 'play_end']

df_ball_snap = util.load_csv('./pass_result_probs/','{}_pass_result_probabilities.csv'.format('ball_snap')) 
df_pass_forward = util.load_csv('./pass_result_probs/','{}_pass_result_probabilities.csv'.format('pass_forward')) 
df_pass_arrived = util.load_csv('./pass_result_probs/','{}_pass_result_probabilities.csv'.format('pass_arrived')) 

df_pass_forward = pd.merge(df_pass_forward, df_plays_teams, how="left", on=["gameId", "playId", "passResult"])
df_pass_arrived = pd.merge(df_pass_arrived, df_plays_teams, how="left", on=["gameId", "playId", "passResult"])
df_ball_snap = pd.merge(df_ball_snap, df_plays_teams, how="left", on=["gameId", "playId", "passResult"])

merge_keys = ['passResult', 'gameId', 'playId','playDescription', 'quarter', 'down',
       'yardsToGo', 'possessionTeam', 'playType', 'yardlineSide',
       'yardlineNumber', 'offenseFormation', 'personnelO', 'defendersInTheBox',
       'numberOfPassRushers', 'personnelD', 'typeDropback',
       'preSnapVisitorScore', 'preSnapHomeScore', 'gameClock',
       'absoluteYardlineNumber', 'penaltyCodes', 'penaltyJerseyNumbers',
       'offensePlayResult', 'playResult', 'epa', 'isDefensivePI', 'gameDate',
       'gameTimeEastern', 'homeTeamAbbr', 'visitorTeamAbbr', 'week',
       'defenseTeam']

df = pd.merge(df_ball_snap, df_pass_arrived, how="inner", on=merge_keys)
df = pd.merge(df, df_pass_forward, how="inner", on=merge_keys)


pass_forward_mean = df[['p_Incomplete pass_forward', 'defenseTeam']].groupby('defenseTeam').describe()['p_Incomplete pass_forward']['mean']
pass_forward_mean = pass_forward_mean - np.mean(pass_forward_mean)



ipdb.set_trace()

possession_team = util.get_slice_by_id(df_plays, playId, gameId)['possessionTeam']

offense, defense = util.get_offense_defense()
