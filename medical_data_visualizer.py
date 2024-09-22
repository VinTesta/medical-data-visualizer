import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where(
    (df['weight'] / ((df['height'] / 100) ** 2)) > 25, 
    1, 0)

# 3
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        df, 
        id_vars=['cardio'], 
        value_vars=[
            'cholesterol', 
            'gluc', 
            'smoke', 
            'alco', 
            'active', 
            'overweight'
        ]
    )
    
    # 6
    
    # 7
    fig = sns.catplot(
        x='variable',
        hue='value', 
        col='cardio', 
        data=df_cat,  
        kind='count',  
        height=5,  
        aspect=1.2,  
        order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )
    fig.set_ylabels('total')
    
    # 8
    fig.fig.subplots_adjust(top=0.9)
    fig.fig.suptitle('Contagem de Valores Categóricos por Variável e Cardio')

    # 9
    fig.savefig('catplot.png')
    
    return fig.fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(corr)

    # 14
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    sns.heatmap(
        corr, 
        annot=True, 
        fmt='.1f', 
        linewidths=1, 
        mask=mask, 
        square=True, 
        cbar_kws={
            'shrink': 0.5
        }
    )


    # 16
    fig.savefig('heatmap.png')
    return fig
