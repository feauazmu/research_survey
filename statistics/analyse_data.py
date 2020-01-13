import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels
from sklearn.datasets import load_iris

from statsmodels.iolib.summary2 import summary_col

#parameters
vignette_names = ['undefined', 'zsg', 'non-zsg']
vignettes = ['undefined', 'zsg', 'nzsg']
colors = ['#bc5090','#58508d','#FF6361' ]
resolution = 1200
ci_value = 1.96 #95% CI normal dist. z+ value

#load data
df = pd.read_csv('cleaned_data.csv')

#create useful columns
df['counter'] = 1
df['diff_zs_nzs'] = df['zsg_question_3'] - df['nzsg_question_3']

df['avg_redist_amount'] = df[['nzsg_question_3', 'zsg_question_3', 'undefined_question_3']].mean(axis=1)

df['zero_redist'] = np.where(df['zsg_question_3'] == df['nzsg_question_3'], 1, 0)
df['zero_redist1'] = np.where(df['undefined_question_3'] == df['zsg_question_3'], 1, 0)

#df['zero_redist2'] = np.where(df['undefined_question_3'] == df['nzsg_question_3'], 1, 0)   #consistency check
#df['zero_redist2'].sum()

df['zero_redist'] = np.where(df['zero_redist'] + df['zero_redist1'] == 2, 1, 0)
redist1 = df[df['zero_redist'] == 1 ]


#drop less interesting columns
drop_cols = []
drop_cols = [c for c in df.columns if c.lower()[:4] in ['bzsg', 'redi']]
drop_cols.remove('bzsg_factor')
drop_cols.remove('redist_factor')

drop_cols.append('Unnamed: 0')
drop_cols.append('zero_redist1')
df = df.drop(columns=drop_cols)

#normalize factors
df['bzsg_factor']= -df['bzsg_factor']/3.887762
df['redist_factor'] = df['redist_factor']/1.871011
#normalize question 1 fairness
for vign in vignettes:
    df[vign + '_question_1'] = (df[vign + '_question_1'] - 1)/6

#stata code
controls =  'age + C(gender) + C(education) + C(field_of_work) + C(nationality)+ C(treatment) + C(political_party)'
#redist_factor on bzsg_factor
mod = smf.ols(formula='redist_factor ~ bzsg_factor', data=df)
res1 = mod.fit()


mod = smf.ols(formula='redist_factor ~ ' + controls + ' + bzsg_factor', data=df)
res2 = mod.fit()

textfile = open('statistics/output/regressions/redist_on_bzsg_factors.txt', 'w')
textfile.write(summary_col([res1, res2],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)}).as_latex())

textfile.close()


#redistributed amount each vignette
for vign in vignettes:
    mod = smf.ols(formula=vign + '_question_3 ~ bzsg_factor', data=df)
    res1 = mod.fit()
    mod = smf.ols(formula=vign + '_question_3 ~ bzsg_factor + redist_factor', data=df)
    res2 = mod.fit()
    mod = smf.ols(formula=vign + '_question_3 ~ bzsg_factor + redist_factor + ' + vign + '_question_1', data=df)
    res3 = mod.fit()
    mod = smf.ols(formula=vign + '_question_3 ~ bzsg_factor + redist_factor + ' + vign + '_question_1 + ' + controls , data=df)
    res4 = mod.fit()
    textfile = open('statistics/output/regressions/redist_amoun_on_factors_' + vign + '.txt', 'w')
    textfile.write(summary_col([res1, res2, res3, res4],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)','\n(2)','\n(3)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'R2':lambda x: "{:.2f}".format(x.rsquared)}).as_latex())
    print(summary_col([res1, res2, res3, res4],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)','\n(2)','\n(3)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'R2':lambda x: "{:.2f}".format(x.rsquared)}))
    textfile.close()



mod = smf.ols(formula='redist_factor ~ ' + ' + bzsg_factor', data=df)
res2 = mod.fit()




#redistributed amount on
mod = smf.ols(formula='nzsg_question_3 ~ '+ controls  +   ' + nzsg_question_1 + redist_factor + bzsg_factor ', data=df)

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
sm.add_constant(df[[ 'bzsg_factor']])

#redistributed amount means

#Vignette: Fairness
mean = df[['undefined_question_1', 'zsg_question_1', 'nzsg_question_1']].mean()
std = df[['undefined_question_1', 'zsg_question_1', 'nzsg_question_1']].std()
error = std*ci_value/np.sqrt(len(df))

fig, mra = plt.subplots()
mra.bar(vignette_names, mean, color=colors, yerr=error, align='center', alpha=0.75, ecolor='black', capsize=10)

mra.set_title('Vignettes: Fairness')
mra.set_ylabel('How unfair')
mra.set_xticklabels(vignette_names, rotation=0)

mra.get_figure().savefig('statistics/output/graphs/Vignettes_Fairness.png', dpi=resolution)




#Vignttes want to redistribute
mean = df[['undefined_question_2', 'zsg_question_2', 'nzsg_question_2']].mean()
std = df[['undefined_question_2', 'zsg_question_2', 'nzsg_question_2']].std()
error = std*ci_value/np.sqrt(len(df))

fig, mra = plt.subplots()
mra.bar(vignette_names, mean, color=colors, yerr=error, align='center', alpha=0.75, ecolor='black', capsize=10)

mra.set_title('Vignettes: Redistribute')
mra.set_ylabel('Fraction')
mra.set_xticklabels(vignette_names, rotation=0)

mra.get_figure().savefig('statistics/output/graphs/Vignettes_Redistribute.png', dpi=resolution)


#Vignettes mean redistributed amount
mean = df[['undefined_question_3', 'zsg_question_3', 'nzsg_question_3']].mean()
std = df[['undefined_question_3', 'zsg_question_3', 'nzsg_question_3']].std()
error = std*ci_value/np.sqrt(len(df))

fig, mra = plt.subplots()
mra.bar(vignette_names, mean, color=colors, yerr=error, align='center', alpha=0.75, ecolor='black', capsize=10)

mra.set_title('Vignettes: Mean Redistributed Amount')
mra.set_ylabel('Fraction')
mra.set_xticklabels(vignette_names, rotation=0)

mra.get_figure().savefig('statistics/output/graphs/Vignettes_Mean_Redistributed_Amount.png', dpi=resolution)

#Vignettes Updated redistributed amount
mean = df[['undefined_question_4', 'zsg_question_4', 'nzsg_question_4']].mean()
std = df[['undefined_question_4', 'zsg_question_4', 'nzsg_question_4']].std()
error = std*ci_value/np.sqrt(len(df))

fig, mra = plt.subplots()
mra.bar(vignette_names, mean, color=colors, yerr=error, align='center', alpha=0.75, ecolor='black', capsize=10)

mra.set_title('Vignettes: Updated Mean Redistributed Amount')
mra.set_ylabel('Fraction')
mra.set_xticklabels(vignette_names, rotation=0)

mra.get_figure().savefig('statistics/output/graphs/Vignettes_Updated_Mean_Redistributed_Amount.png', dpi=resolution)


#Distribution of reddistributed amount
for name in vignettes:
    b = df[df[name+'_question_2'] == 1][name+'_question_3'].plot.hist(bins=16)
    b.set_title('Redistrubted Amount ' + name)
    b.set_xlabel('Fraction redistributed')
    b.get_figure().savefig('statistics/output/graphs/' + name + '_DistributionOfRedistAmount.png', dpi=resolution)
    plt.clf()






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

ax = sns.regplot(y='nredist_avg', x='nbzsg_avg', data=df).get_figure().savefig('statistics/output/graphs/redist_on_bzsg_avg.png', dpi=resolution)

ax1 = sns.regplot(y='redist_factor', x='bzsg_factor', data=df).get_figure().savefig('statistics/output/graphs/redist_on_bzsg_factors.png', dpi=resolution)

ax2 = sns.regplot(y='zsg_question_3', x='nredist_avg', data=df)

df.plot(kind='scatter',x='age',y='bzsg_factor',color='red')
plt.show()

sns.lmplot("age", "bzsg_factor", data=df, hue="nationality", fit_reg=False, col='gender', col_wrap=2)
df.corr(method='pearson')
