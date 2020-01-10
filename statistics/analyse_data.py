import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import statsmodels.formula.api as smf

from sklearn.datasets import load_iris



df = pd.read_csv('cleaned_data.csv')

#create useful columns
df['counter'] = 1
df['diff_zs_nzs'] = df['zsg_question_3'] - df['nzsg_question_3']
df.columns
df['avg_redist_amount'] = df[['nzsg_question_3', 'zsg_question_3', 'undefined_question_3']].mean(axis=1)

df['zero_redist'] = np.where(df['zsg_question_3'] == df['nzsg_question_3'], 1, 0)
df['zero_redist1'] = np.where(df['undefined_question_3'] == df['zsg_question_3'], 1, 0)

#df['zero_redist2'] = np.where(df['undefined_question_3'] == df['nzsg_question_3'], 1, 0)   #consistency check
#df['zero_redist2'].sum()

df['zero_redist'] = np.where(df['zero_redist'] + df['zero_redist1'] == 2, 1, 0)
redist1 = df[df['zero_redist'] == 1 ]
redist1[redist1['zsg_question_3'] != 0]
df.groupby('zero_redist').mean()

#drop less interesting columns
drop_cols = []
drop_cols = [c for c in df.columns if c.lower()[:4] in ['bzsg', 'redi']]
drop_cols.remove('bzsg_factor')
drop_cols.remove('redist_factor')

drop_cols.append('Unnamed: 0')
drop_cols.append('zero_redist1')
df = df.drop(columns=drop_cols)



#stata code
mod = smf.ols(formula='avg_redist_amount ~ bzsg_factor', data=df)

res = mod.fit()
print(res.summary())



#averages and correlations

df.groupby('gender').mean()
df.groupby('nationality').mean()
df.groupby('age').mean()

df.groupby('undefined_question_1').count().plot(kind='bar', y='counter', color='blue')


df.groupby('zsg_question_1').count().plot(kind='bar', y='counter', color='blue')


df.groupby('undefined_question_2').count().plot(kind='bar', y='counter', color='blue')
df.groupby('zsg_question_2').count().plot(kind='bar', y='counter', color='blue')

df.groupby('nzsg_question_2').count().plot(kind='bar', y='counter', color='blue')


df.groupby('zero_redist')['avg_redist_amount'].plot.hist(bins=10)
df.plot(kind='hist',x='undefined_question_3',color='blue')

df['diff_zs_undef'] = df['zsg_question_3'] - df['undefined_question_3']

df.plot(kind='scatter',x='diff_zs_undef', y='bzsg_factor',color='blue')


df.groupby('political_party').sum()

import seaborn as sns; sns.set(color_codes=True)

ax = sns.regplot(y='nredist_avg', x='nbzsg_avg', data=df)

ax1 = sns.regplot(y='redist_factor', x='bzsg_factor', data=df)

ax2 = sns.regplot(y='zsg_question_3', x='nredist_avg', data=df)

df.plot(kind='scatter',x='age',y='bzsg_factor',color='red')
plt.show()

sns.lmplot("age", "bzsg_factor", data=df, hue="nationality", fit_reg=False, col='gender', col_wrap=2)
df.corr(method='pearson')
