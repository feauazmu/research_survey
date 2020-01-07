import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo


df = pd.read_csv('cleaned_data.csv')
df['counter'] = 1

#averages and correlations


df.groupby('inequality_research_survey.1.player.gender').mean()
df.groupby('inequality_research_survey.1.player.nationality').mean()
df.groupby('inequality_research_survey.1.player.age').mean()

df.groupby('inequality_research_survey.1.player.undefined_question_1').count().plot(kind='bar', y='counter', color='blue')

df.mean()
df.groupby('inequality_research_survey.1.player.zsg_question_1').count().plot(kind='bar', y='counter', color='blue')


df.groupby('inequality_research_survey.1.player.undefined_question_2').count().plot(kind='bar', y='counter', color='blue')
df.groupby('inequality_research_survey.1.player.zsg_question_2').count().plot(kind='bar', y='counter', color='blue')

df.groupby('inequality_research_survey.1.player.nzsg_question_2').count().plot(kind='bar', y='counter', color='blue')

df.mean().plot()

df.plot(kind='hist',x='inequality_research_survey.1.player.undefined_question_3',color='blue')

df['diff_zs_undef'] = df['inequality_research_survey.1.player.zsg_question_3'] - df['inequality_research_survey.1.player.undefined_question_3']

df.plot(kind='scatter',x='diff_zs_undef', y='bzsg_factor',color='blue')


df.groupby('inequality_research_survey.1.player.political_party').sum()

import seaborn as sns; sns.set(color_codes=True)

ax = sns.regplot(y='nredist_avg', x='nbzsg_avg', data=df)

ax1 = sns.regplot(y='redist_factor', x='bzsg_factor', data=df)

ax2 = sns.regplot(y='inequality_research_survey.1.player.zsg_question_3', x='nredist_avg', data=df)

df.plot(kind='scatter',x='inequality_research_survey.1.player.age',y='bzsg_factor',color='red')
plt.show()

sns.lmplot("inequality_research_survey.1.player.age", "bzsg_factor", data=df, hue="inequality_research_survey.1.player.nationality", fit_reg=False, col='inequality_research_survey.1.player.gender', col_wrap=2)
df.corr(method='pearson')
