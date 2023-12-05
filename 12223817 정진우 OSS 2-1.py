import pandas as pd
import numpy as np
from pandas import Series, DataFrame


data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

def print_top_10_players():
    year = 2014

    for i in range(4):
        year = year + 1
       
        H10 = data_df[data_df['year'] == year].sort_values(by='H', ascending=False)['batter_name'].head(10).to_string(index=False)
        avg10 = data_df[data_df['year'] == year].sort_values(by='avg', ascending=False)['batter_name'].head(10).to_string(index=False)
        HR10 = data_df[data_df['year'] == year].sort_values(by='HR', ascending=False)['batter_name'].head(10).to_string(index=False)
        OBP10 = data_df[data_df['year'] == year].sort_values(by='OBP', ascending=False)['batter_name'].head(10).to_string(index=False)
    
        print(year,'년도 각 상위 10위 선수들','\n',sep='')
        print('*안타*\n',H10,sep='\n\n')
        print('*타율*',avg10,sep='\n\n')
        print('*홈런*',HR10,sep='\n\n')
        print('*출루율*',OBP10,sep='\n\n')
    
def print_war_top10_in_2018():
    year = 2018

    positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

    for position in positions:
        players = data_df[(data_df['year'] == year) & (data_df['cp'] == position)].sort_values(by='war', ascending=False)['batter_name']
        print('*',position,'*',sep='')
        print(players.to_string(index=False),'\n',sep='')
        

def print_highest_correlation():
    Corr= data_df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']].corrwith(data_df['salary'])
    print(Corr.idxmax())
    
      

if __name__=='__main__':
    print('**2015~2018년도 각 분야별 상위 10위 선수들**','\n')
    print_top_10_players()
    print('\n','**2018도 각 포지션별 승리기여도가 높은 선수들**','\n')
    print_war_top10_in_2018()
    print('\n','**연봉과 가장 상관관계가 높은 분야**','\n')
    print_highest_correlation()