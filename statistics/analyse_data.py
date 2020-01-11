import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels
from sklearn.datasets import load_iris



df = pd.read_csv('cleaned_data.csv')
vignette_names = ['undefined', 'zsg', 'non-zsg']

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

#drop less interesting columns
    drop_cols = []
drop_cols = [c for c in df.columns if c.lower()[:4] in ['bzsg', 'redi']]
drop_cols.remove('bzsg_factor')
drop_cols.remove('redist_factor')
df
drop_cols.append('Unnamed: 0')
drop_cols.append('zero_redist1')
df = df.drop(columns=drop_cols)
df['bzsg_factor'].mean()
df['nredist_avg'].min()
df['bzsg_factor']= -df['bzsg_factor']/3.887762
df['redist_factor'] = df['redist_factor']/1.871011
#stata code
mod = smf.ols(formula='undefined_question_3 ~ bzsg_factor', data=df)

res = mod.fit()
print(res.summary())


mod = smf.ols(formula='redist_factor ~ age + C(gender) + C(education) + C(field_of_work) + C(nationality) + C(political_party) +bzsg_factor ', data=df)

res = mod.fit()
print(res.summary())
mod = smf.ols(formula='nzsg_question_3 ~ nbzsg_avg ', data=df)

res = mod.fit()
print(res.summary())

mod = smf.ols(formula='nzsg_question_3 ~ bzsg_factor ', data=df)

res = mod.fit()
print(res.summary())

logit = sm.Logit(df['undefined_question_2'], sm.add_constant(df[[ 'bzsg_factor']]))
res1 = logit.fit()
print(res1.summary())
df['undefined_question_2'].mean()
sm.add_constant(df[[ 'bzsg_factor']])
df['undefined_question_1'].max()
#redistributed amount means

#Vignette: Fairness
ax = df[['undefined_question_1', 'zsg_question_1', 'nzsg_question_1']].mean().plot(kind='bar', y='counter', color=['m', 'blue', 'red' ])
ax.set_title('Vignettes: Fairness')
ax.set_ylabel('How fair')
ax.set_xticklabels(vignette_names, rotation=0)
#ax.set(xlim=(-1, 3), ylim=(1, 4))
ax.figure
ax.get_figure().savefig('statistics/output/graphs/Vigenttes_Fairness.png')

#Vignttes want to redistribute
ax1 = df[['undefined_question_2', 'zsg_question_2', 'nzsg_question_2']].mean().plot(kind='bar', y='counter', color=['m', 'blue', 'red' ])
ax1.set_title('Vignettes: Redistribute')
ax1.set_ylabel('Fraction')
ax1.set_xticklabels(vignette_names, rotation=0)
#ax1.set(xlim=(-1, 3), ylim=(1, 4))
ax.figure
ax1.get_figure().savefig('statistics/output/graphs/Vigenttes_Redistribute.png')

#Vignettes mean redistributed amount
ax2 = df[['undefined_question_3', 'zsg_question_3', 'nzsg_question_3']].mean().plot(kind='bar', color=['m', 'blue', 'red' ])
ax2.set_title('Vignettes: Mean Redistributed Amount')
ax2.set_ylabel('Fraction')
ax2.set_xticklabels(vignette_names, rotation=0)
#ax1.set(xlim=(-1, 3), ylim=(1, 4))
ax2.figure
ax2.get_figure().savefig('statistics/output/graphs/Vigenttes_Mean_Redistributed_Amount.png')

#Vignettes Updated redistributed amount
ax3 = df[['undefined_question_4', 'zsg_question_4', 'nzsg_question_4']].mean().plot(kind='bar', color=['m', 'blue', 'red' ])
ax3.set_title('Vignettes: Mean Redistributed Amount')
ax3.set_ylabel('Fraction')
ax3.set_xticklabels(vignette_names, rotation=0)
#ax1.set(xlim=(-1, 3), ylim=(1, 4))
ax3.figure
ax3.get_figure().savefig('statistics/output/graphs/Vigenttes_Updated_Mean_Redistributed_Amount.png')

type(ax)
df.mean()
df[['undefined_question_3', 'zsg_question_3', 'nzsg_question_3']].mean().plot(kind='bar', y='counter', color=['m', 'blue', 'red' ])
df[['undefined_question_4', 'zsg_question_4', 'nzsg_question_4']].mean().plot(kind='bar', y='counter', color=['m', 'blue', 'red' ], label =['a', 'b', 'c'])

redistributed amount distribution
df[df['undefined_question_2'] == 1]['undefined_question_3'].plot.hist(bins=8)
df[df['zsg_question_2'] == 1]['zsg_question_3'].plot.hist(bins=7)
df[df['nzsg_question_2'] == 1]['nzsg_question_3'].plot.hist(bins=7)
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
