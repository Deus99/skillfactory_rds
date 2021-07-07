import pandas as pd
import numpy as np
import streamlit as st
import numpy as np
import os
import math

import matplotlib.pyplot as plt
color_map = plt.cm.winter
from matplotlib.patches import RegularPolygon
from matplotlib.patches import Arc, Circle, Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
c = mcolors.ColorConverter().to_rgb
positive_cm = ListedColormap([c('#e1e5e5'),c('#e78c79'),c('#d63b36')]) # Positive map
negative_cm = ListedColormap([c('#e1e5e5'), c('#a1ceee'),c('#28aee4')]) # Negative map

from PIL import Image


def create_rink():
    """
    Create hockey rink in matplotlib (by blueprint)
    """
    
    fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
    # Нейтральная зона
    # Центральная линия
    line = plt.Line2D((0, 0), (-42.5, 42.5), lw=5, color='red', linestyle='-')
    plt.gca().add_line(line)

    line = plt.Line2D((0, 0), (-42.5, 42.5), lw=2, color='white', linestyle='--')
    plt.gca().add_line(line)

    # синяя линия
    line = plt.Line2D((25, 25), (-42.5, 42.5), lw=5, color='blue', linestyle='-')
    plt.gca().add_line(line)

    # Центральный круг
    ax.add_patch(Arc((0, 0), 30, 30, theta1=-90, theta2=90, lw=2, edgecolor='blue'))
    ax.add_patch(Circle((0, 0), 1.5, lw=2.5, edgecolor='blue', facecolor='blue'))

    # точки
    ax.add_patch(Circle((20, 22), 1, lw=5, edgecolor='red', facecolor='red'))
    ax.add_patch(Circle((20, -22), 1, lw=5, edgecolor='red', facecolor='red'))

    # Верхний круг вбрасывания
    line = plt.Line2D((75, 71, 71), (23, 23, 26), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((63, 67, 67), (23, 23, 26), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((63, 67, 67), (21, 21, 18), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((75, 71, 71), (21, 21, 18), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)

    line = plt.Line2D((71, 71), (7, 5), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((67, 67), (7, 5), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((67, 67), (37, 39), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((71, 71), (37, 39), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)

    ax.add_patch(Circle((69, 22), 1, lw=5, edgecolor='red', facecolor='red'))
    ax.add_patch(Arc((69, 22), 30, 30, theta1=0, theta2=360, lw=2, edgecolor='red'))
    
    # Нижний круг вбрасывания
    line = plt.Line2D((75, 71, 71), (-23, -23, -26), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((63, 67, 67), (-23, -23, -26), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((63, 67, 67), (-21, -21, -18), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((75, 71, 71), (-21, -21, -18), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)

    line = plt.Line2D((71, 71), (-7, -5), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((67, 67), (-7, -5), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((67, 67), (-37, -39), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((71, 71), (-37, -39), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)

    ax.add_patch(Circle((69, -22), 1, lw=5, edgecolor='red', facecolor='red'))
    ax.add_patch(Arc((69, -22), 30, 30, theta1=0, theta2=360, lw=2, edgecolor='red'))


    #Зона ворот
    line = plt.Line2D((89, 89), (-40.7, 40.7), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    ax.add_patch(Arc((89, 0), 16, 16, theta1=90, theta2=270, lw=2, edgecolor='red', facecolor='blue'))
    ax.add_patch(Rectangle((85.5,-4), 3.5, 8, lw=2 ,edgecolor='red', facecolor='blue', alpha=0.7))

    ax.add_patch(Arc((90, 1), 4, 4, theta1=-30, theta2=90, lw=2, edgecolor='red', facecolor='blue'))
    ax.add_patch(Arc((90, -1), 4, 4, theta1=270, theta2=30, lw=2, edgecolor='red', facecolor='blue'))
    line = plt.Line2D((89, 90), (3, 3), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)
    line = plt.Line2D((89, 90), (-3, -3), lw=2, color='red', linestyle='-')
    plt.gca().add_line(line)


    # Борта
    line = plt.Line2D((0, 80), (-42.6, -42.6), lw=5, color='black')
    plt.gca().add_line(line)

    line = plt.Line2D((0, 80), (42.6, 42.6), lw=5, color='black')
    plt.gca().add_line(line)

    line = plt.Line2D((100, 100), (-22.6, 22.6), lw=5, color='black')
    plt.gca().add_line(line)

    ax.add_patch(Arc((80, 22.6), 40, 40,
                 theta1=0, theta2=90, edgecolor='black', lw=5))
    ax.add_patch(Arc((80, -22.6), 40, 40,
                 theta1=270, theta2=360, edgecolor='black', lw=5))

    plt.xlim(0, 120)

    #plt.axis('auto')
    #plt.show()
    return ax, fig

def viz_player_shots(title, player='all', seasons=['2019'], missed=False, 
                     passing=False, stren='all', events=['Goal', 'Shot'], 
                     filter=1, shooting=True):
    """
    Стороит карту событий на половине арены. Только нормализованные события (X>0)
    
    player - выбор игрока, по умолчанию 'all' - все игроки
    
    seasons - список сезонов, по умолчанию 2019 - последний полный сезон    

    stren - составы все, even - только равные, pp - только большинство
    
    events - события, по умолчанию ['Goal', 'Shot'], плотность голов и бросков. 
    events = ['Hit'] - плотность силовых приемов или другого события 
    
    missed = True считать броски мимо обычными бросками, по умолчанию False
    
    passing = True, показывает на схеме, голы, которые были забиты с паса конкретного игрока 
    совершенно случайно показывает карту набранных очков за сезон (гол + пас). Броски по воротам не показывают
    
    filter = 1, минимальное значение в одной ячейке, чтобы отфильтровать случайные события, которые были 1 раз 
    
    passing = False - не показывать голы, в которых игрок был ассистентом 
    
    shooting = False  - убирает голы 
    *Passng=True + Shooting=False - показывает только голевые передачи
    
    """
    
    GRIDSIZE = 30
    MINCNT = 0
    
    xbnds = np.array([-100.,100.0])
    ybnds = np.array([-100,100])
    EXTENT = [xbnds[0],xbnds[1],ybnds[0],ybnds[1]]
    
    if seasons=='all':
        seasons = ['2015', '2016', '2017', '2018', '2019', '2020']
    if seasons == 'last':
        seasons = ['2020']
    
    
    
    if player=='all':
        player_data = df.loc[(df.season.isin(seasons))][['player', 'event', 'x', 'y', 'team_str', 'pass1', 'pass2']]
    elif passing:
        player_data = df.loc[(df.season.isin(seasons))&((df.player==player)|(
            df.pass1==player)|(df.pass2==player))][['player', 'event', 'x', 'y', 'team_str', 'pass1', 'pass2']]
    else:
        player_data = df.loc[(df.season.isin(seasons))&(df.player==player)][['player', 
                                                                             'event', 'x', 'y', 'team_str', 'pass1', 'pass2']]
    
    
    
    #normalize
    player_data.loc[player_data.x<0, 'y'] = player_data.loc[player_data.x<0, 'y']*(-1)
    player_data.loc[player_data.x<0, 'x'] = player_data.loc[player_data.x<0, 'x']*(-1)
    
    
    if stren=='even':
        player_data = player_data.loc[player_data.team_str=='Even']
    
    elif stren=='pp':
        player_data = player_data.loc[player_data.team_str=='Power Play']
        
    player_data.loc[player_data.x>96, 'x'] = 96
    player_data.loc[player_data.y>38, 'y'] = 38
    player_data.loc[player_data.y<-38, 'y'] = -38
    
    player_data.loc[(player_data.x>88)&(player_data.y>30), 'x'] = player_data.loc[
        (player_data.x>90)&(player_data.y>30), 'x'] - 5
    player_data.loc[(player_data.x>88)&(player_data.y>30), 'y'] = player_data.loc[
        (player_data.x>90)&(player_data.y>30), 'y'] - 5
    
    player_data.loc[(player_data.x>88)&(player_data.y<-30), 'x'] = player_data.loc[
        (player_data.x>90)&(player_data.y<-30), 'x'] - 5
    
    player_data.loc[(player_data.x>88)&(player_data.y<-30), 'y'] = player_data.loc[
        (player_data.x>90)&(player_data.y<-30), 'y'] - 5
    
    if missed: # только для missed = True , иначе shots = events
        player_x_all_shots = player_data.loc[(player_data.event.isin(['Goal', 'Shot', 'Missed Shot'])&(
            player_data.player==player)), 'x'].tolist()
        player_y_all_shots = player_data.loc[(player_data.event.isin(['Goal', 'Shot', 'Missed Shot'])&(
            player_data.player==player)), 'y'].tolist()
    else:
        player_x_all_shots = player_data.loc[(player_data.event.isin(events))&
                                              (player_data.player==player), 'x'].tolist()
        player_y_all_shots = player_data.loc[(player_data.event.isin(events))&
                                              (player_data.player==player), 'y'].tolist()

# If we need to flip the x coordinate then we need to also flip the y coordinate!
    player_x_all_shots_normalized = player_x_all_shots
    player_y_all_shots_normalized = player_y_all_shots

    if 'Goal' in events:
        player_x_all_goals = player_data.loc[(player_data.event.isin(['Goal']))&(player_data.player==player)]['x'].tolist()
        player_y_all_goals = player_data.loc[(player_data.event.isin(['Goal']))&(player_data.player==player)]['y'].tolist()

        player_x_goal_normalized = player_x_all_goals
        player_y_goal_normalized = player_y_all_goals
    
    if passing:
        player_x_pass = player_data.loc[(player_data.pass1==player)|(player_data.pass2==player)]['x'].tolist()
        player_y_pass = player_data.loc[(player_data.pass1==player)|(player_data.pass2==player)]['y'].tolist()


    ax, fig = create_rink()
        
        
    player_hex_data = ax.hexbin(player_x_all_shots_normalized,player_y_all_shots_normalized,gridsize=GRIDSIZE,
                                 extent=EXTENT,mincnt=MINCNT,alpha=0);
    player_verts_shots = player_hex_data.get_offsets();
    player_shot_frequency = player_hex_data.get_array();
    
    if 'Goal' in events:
        player_goal_hex_data = ax.hexbin(player_x_goal_normalized,
                                      player_y_goal_normalized,gridsize=GRIDSIZE,extent=EXTENT,mincnt=MINCNT,alpha=0.0)
        player_verts_goals = player_goal_hex_data.get_offsets();
        player_goal_frequency = player_goal_hex_data.get_array();
    
    if passing:
        player_assists_data = ax.hexbin(player_x_pass,player_y_pass,gridsize=GRIDSIZE,
                                 extent=EXTENT,mincnt=MINCNT,alpha=0);
        player_verts_pass = player_assists_data.get_offsets();
        player_pass_frequency = player_assists_data.get_array();

    
    width=100
    height=84
    #scalingx=width/100-0.6;
    scalingx=1;
    #scalingy=height/100+0.5;
    scalingy=1;
    #scalingy=10
    x_trans=0;
    y_trans=0
    S = 3.15*scalingx;

# Loop over the locations and draw the hex
    
    if passing:
        for i,v in enumerate(player_verts_pass):  
            if player_pass_frequency[i] < filter: continue
            
            scaled_player_pass_frequency = player_pass_frequency[i]/max(player_pass_frequency)
            radius = S*math.sqrt(scaled_player_pass_frequency)
             #Scale the radius to the number of goals made in that area
            hex = RegularPolygon((x_trans+v[0]*scalingx, (y_trans+v[1]*scalingy)), \
                         numVertices=6, radius=radius*1.1, orientation=np.radians(0), 
                         facecolor='#FFA500', alpha=1, edgecolor=None)
            ax.add_patch(hex) 
        
    else:
        for i,v in enumerate(player_verts_shots):
            if player_shot_frequency[i] < filter: continue
    
            scaled_player_shot_frequency = player_shot_frequency[i]/max(player_shot_frequency)
            radius = S*math.sqrt(scaled_player_shot_frequency)
            hex = RegularPolygon((x_trans+v[0]*scalingx, y_trans+v[1]*scalingy), \
                         numVertices=6, radius=radius*1.1, orientation=np.radians(0), \
                          facecolor='#CA0020',alpha=0.8, edgecolor=None)
            ax.add_patch(hex)
        
    if ('Goal' in events) and shooting:
        for i,v in enumerate(player_verts_goals):
            if player_goal_frequency[i] < filter: continue
            
            scaled_player_goal_frequency = player_goal_frequency[i]/max(player_goal_frequency)
            radius = S*math.sqrt(scaled_player_goal_frequency)
             #Scale the radius to the number of goals made in that area
            hex = RegularPolygon((x_trans+v[0]*scalingx, (y_trans+v[1]*scalingy)), \
                         numVertices=6, radius=radius*1.1, orientation=np.radians(0), \
                         facecolor='#7CFC00', alpha=1, edgecolor='black')
            ax.add_patch(hex) 
            
        
    plt.title(title)
    plt.xlim(-10, 125);
    plt.ylim(-50,50);
    st.pyplot(fig=plt)

###########STREAMLIT###########

st.set_page_config(
    page_title="Kraken Expansion Draft",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    )   



st.title('KRAKEN DRAFT SIMULATION')



@st.cache(allow_output_mutation=True)
def persistdata():
    return {}


INPUT_DIR = r"./Data/"

pst = pd.read_csv(INPUT_DIR + 'pst.csv')
rt = pd.read_csv(INPUT_DIR + 'ratings.csv')
rnn_pred = pd.read_csv(INPUT_DIR + 'rnn_pred.csv')
drf = pd.read_csv(INPUT_DIR + 'draft_players.csv')
dk = pd.read_csv(INPUT_DIR + 'draft_keepers.csv')
df = pd.read_csv(INPUT_DIR + 'df.csv')
gs = pd.read_csv(INPUT_DIR + 'goalies.csv')
pls = pd.read_csv((INPUT_DIR + 'players.csv'))


## ОБНУЛЕНИЕ
team_drafted = list()
selected_players = persistdata()
all_teams = set(drf.team.unique().tolist()) # все команды в множестве

# последние рейтинги 2019-2020
last_rt = rt.loc[rt.season.between(2019,2020)].drop_duplicates(subset='player', keep='last')

drf.rename({'position': 'field_position'}, axis=1, inplace=True)
drf.loc[drf.player=='Sebastian Aho', 'player'] = 'Sebastian Aho LD'

# список полевых игроков с рейтингами доступных для драфта
draftlist = drf.merge(last_rt, left_on='player', right_on='player', how='left').dropna()

draftlist['caphit'] = draftlist['caphit'].apply(lambda x: x[1::].replace(',','')).astype('int')

# список вратарей
gs = gs.loc[gs.situation=='all', ['name', 'games_played', 'team', 'position', 'icetime', 'xGoals', 'goals']]
gs['xG_goals'] = gs['xGoals'] / gs['goals']
gs.drop(['team', 'position'], axis=1, inplace=True)
draftkeepers = dk.merge(gs, left_on='player', right_on='name').drop(['name', 'decr'], axis=1).replace(np.inf, 0)



### ДОБАВЛЯЕМ RNN В СТАТИСТИКУ ####
pl_stats = pst.groupby(['person', 'season'])['timeOnIce', 'assists', 'goals', 'shots', 
                                  'hits', 'blocked', 'takeaways', 'plusMinus'].sum().reset_index().query('season>2017').drop_duplicates()

pls_st_name = pl_stats.merge(pls[['id', 'fullName']], left_on='person', right_on='id', how='left').drop(
    'id', axis=1).drop_duplicates()

rnn_pred['season'] = 2021
pls_st_name.rename(columns={'fullName': 'player'}, inplace=True)
pls_st_name.drop('person', axis=1, inplace=True)

pls_st_name = pd.concat([pls_st_name, rnn_pred])

pls_st_name['points'] = pls_st_name['goals'] + pls_st_name['assists']
pls_st_name['timeOnIce'].fillna(0, inplace=True)
pls_st_name['timeOnIce'] = pls_st_name['timeOnIce'].astype('int')
####

# списки и фильтры
#field_pos = draftlist['field_position'].unique()
#field_pos = np.append(field_pos, 'goalkeeper')
#player_position = draftlist['position'].unique()



team = draftlist['team'].unique().tolist()
team.append('all')

sel_teams = np.array(None)

##### STREAMLIT  - ВЫБОР КОМАНДЫ ####


sel_team = st.sidebar.selectbox("Выбирите команду", team, index=0) 
sel_position = st.sidebar.selectbox("Выберите позицию", ['forward', 'defense', 'goalkeeper'], index=0)

ufa21 = st.sidebar.checkbox('Показывать UFA(2021)', value=True)



max_age = st.sidebar.slider('Maximum age?', 18, 40, 26)


###### SELECT TEAM  / SELECT POSITION ############
st.header('Доступные для выбора игроки')
if sel_team!='all':
    if sel_position != 'goalkeeper':
        if not ufa21:
            st.dataframe(draftlist.loc[(draftlist.team==sel_team)&(draftlist.expiry!='UFA (2021)')&(
                draftlist.field_position==sel_position)&(draftlist.age<=max_age)], height=300)
        else:
            st.dataframe(draftlist.loc[(draftlist.team==sel_team)&(
                draftlist.field_position==sel_position)&(draftlist.age<=max_age)], height=300)
    
    else:
        if not ufa21:
            st.dataframe(draftkeepers.loc[(draftkeepers.team==sel_team)&(
                draftkeepers.expiry!='UFA (2021)')&(draftkeepers.age<=max_age)], height=300)
        else:
            st.dataframe(draftkeepers.loc[(draftkeepers.team==sel_team)&(
                draftkeepers.age<=max_age)], height=300)

        #player_name = draftlist.loc[(draftlist.team==sel_team), 'player'].unique()
        #st.dataframe(draftlist, height=800)
else:
    show_taken = st.sidebar.checkbox('убрать игроков из драфтованных команд', value=True)
    if show_taken:
        show_teams = all_teams - set(selected_players.keys())
        show_teams = list(show_teams)
    else:
        show_teams = all_teams

    if sel_position != 'goalkeeper':
        if not ufa21:
            st.dataframe(draftlist.loc[(draftlist.expiry!='UFA (2021)')&(
                draftlist.field_position==sel_position)&(draftlist.age<=max_age)&(draftlist.team.isin(show_teams))], height=300)
        else:
            st.dataframe(draftlist.loc[(draftlist.field_position==sel_position)&(
                    draftlist.age<=max_age)&(draftlist.team.isin(show_teams))], height=300)
    else:
        if not ufa21:
            st.dataframe(draftkeepers.loc[(draftkeepers.expiry!='UFA (2021)')&(
                draftkeepers.age<=max_age)&(draftkeepers.team.isin(show_teams))], height=300)
        else:
            st.dataframe(draftkeepers.loc[(draftlist.age<=max_age)&(draftkeepers.team.isin(show_teams))], height=300)

    #player_name = draftlist.loc[(draftlist.expiry!='UFA (2021)'), 'player'].unique()
    #st.dataframe(draftlist, height=800)

##########



##### STREAMLIT  - ВЫБОР ИГРОКА ####
#pl_sel = st.sidebar.multiselect("Выбирите игрока", player_name, )
#sel_teams = draftlist.loc[draftlist.player.isin(pl_sel)].team.unique()

try:
    current_player=selected_players[sel_team]
    if current_player in drf.player.unique().tolist():
        cut_df = draftlist.loc[draftlist.team==sel_team].reset_index()
    else:
        cut_df = draftkeepers.loc[draftkeepers.team==sel_team].reset_index()    
    
    sel_index =  cut_df.loc[cut_df.player==current_player].index[0]
    sel_index = int(sel_index)
except KeyError:
    #sel_index= draftlist.loc[draftlist.team==sel_team].index.min()
    sel_index=0



##### RADIO #####

if sel_position != 'goalkeeper':
    select_player = st.sidebar.radio('Выбирите игрока', draftlist.loc[draftlist.team==sel_team, 'player'].tolist(), index = sel_index)
    #take_team = draftlist.loc[draftlist.player==take_player, 'team'].tolist()[0]
    #selected_players[sel_team] = take_player
else:
    select_player = st.sidebar.radio('Выбирите игрока', draftkeepers.loc[draftkeepers.team==sel_team, 'player'].tolist(), index = sel_index)

############



###### TAKE PLAYER ##########

take_player = st.sidebar.button('Взять игрока')

############################


if take_player:
    selected_players[sel_team] = select_player
    



#if not ufa21:
#    st.dataframe(draftlist.loc[(draftlist.team==sel_team)&(draftlist.expiry!='UFA (2021)')], height=300)
#else:
#    st.dataframe(draftlist.loc[(draftlist.team==sel_team)], height=300)




select_df = drf.merge(last_rt, left_on='player', right_on='player', how='left')
select_df = select_df.loc[select_df.player.isin(selected_players.values())].dropna()
select_gk = dk.loc[dk.player.isin(selected_players.values())]


def calculate_caphit(df):
    df['caphit'] = df['caphit'].apply(lambda x: x.replace('$', '').replace(',', ''))
    df['caphit'] = df['caphit'].astype('int')
    df = df.loc[df.expiry!='RFA (2021)']

    return (df['caphit'].sum() / 1000000).round(1)

try: 
    st.sidebar.write("выбрано нападающих: ", select_df.field_position.value_counts()['forward'], ' / 14')
except KeyError:
    st.sidebar.write("Выбрано нападающих: 0 / 14")

try:
    st.sidebar.write("выбрано защитников: ", select_df.field_position.value_counts()['defense'], ' / 9')
except KeyError:
    st.sidebar.write("выбрано защитников: 0 / 9")

try:
    st.sidebar.write("выбрано вратарей: ", len(select_gk), ' / 3')
except KeyError:
    st.sidebar.write("выбрано вратарей: 0 / 3")

st.sidebar.write("Суммарная платежная ведомость на сезон 21/22:", calculate_caphit(select_df) + calculate_caphit(select_gk), 'из 48,9 млн' )

st.header('Выбранные полевые игроки')

st.write(select_df.drop_duplicates())

st.header('Выбранные вратари')
st.write(select_gk.drop_duplicates())

select_df.to_csv('selected_player.csv', index=False)
select_gk.to_csv('selected_keepers.csv', index=False)



events_list = ['goals and passes', 'goals and shots', 'hits', 'takeaways']
stren_list = ['all', 'even', 'pp']
seasons_list = [2019, 2020]

events = 'goals and shoots'
passing_tf = False
shooting_tf = True
events_sel = ['Goal', 'Shot']
stren_sel = 'all'


##### VIZ HEATMAP  #### 
st.header("Информация об игроке")
st.write('2021 - прогноз статистики на сезон 2021-2022')
stats_sel_player = pls_st_name.loc[pls_st_name.player==select_player]
st.write(stats_sel_player.drop('player', axis=1)[['season', 'goals', 'assists', 'points', 'shots', 'hits', 'blocked', 'plusMinus']])


stren_sel = st.selectbox("Выберите составы команд, равные или большинство", stren_list, index=0)
events = st.selectbox("Выберете показатель", events_list, index=1)
seasons_sel = st.multiselect("Выберите сезоны: ", seasons_list, default=2020)

if events == 'goals and passes':
    passing_tf = True
    shooting_tf = True
    events_sel = ['Goal', 'Pass']

elif events == 'goals and shoots':
    passing_tf = False
    shooting_tf = True
    events_sel = ['Goal', 'Shot']

elif events == 'hits':
    passing_tf = False
    shooting_tf = False
    events_sel = ['Hit']

elif events == 'takeaways':
    passing_tf = False
    shooting_tf = False
    events_sel = ['Takeaway']

viz_title = events + ' ' + str(select_player) + ' сезоны 19-20 и 20-21'

viz_player_shots(viz_title, player=select_player, seasons=seasons_sel, missed=False, 
                     passing=passing_tf, stren=stren_sel, events=events_sel, 
                     filter=1, shooting=shooting_tf)

st.write(selected_players)