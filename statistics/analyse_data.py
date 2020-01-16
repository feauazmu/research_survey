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

df['zsg_effort_diff']
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
                             'R2':lambda x: "{:.2f}".format(x.rsquared)}).as_latex())
    textfile.close()

#redistributed amount on
mod = smf.ols(formula='undefined_question_3 ~ '+ controls  +   ' + nzsg_question_1 + redist_factor + bzsg_factor ', data=df)

res = mod.fit()
print(res.summary())
mod = smf.ols(formula='undefined_question_3 ~ zsg_question_3 ', data=df)

res1 = mod.fit()
print(res.summary())

mod = smf.ols(formula='undefined_question_3 ~ nzsg_question_3 ', data=df)

res2 = mod.fit()
print(res.summary())

mod = smf.ols(formula='undefined_question_3 ~ nzsg_question_3  + zsg_question_3', data=df)

res3 = mod.fit()
print(res.summary())

mod = smf.ols(formula='undefined_question_3 ~ nzsg_question_3  + zsg_question_3 + ' + controls, data=df)

res4 = mod.fit()
print(res3.summary())

textfile = open('statistics/output/regressions/redist_amoun_on_redist_amounts.txt', 'w')
textfile.write(summary_col([res1, res2, res3, res4],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)','\n(2)','\n(3)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)}).as_latex())
print(summary_col([res1, res2, res3, res4],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)','\n(2)','\n(3)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)}).as_latex())
textfile.close()

#logit prob of redistribute
logit = sm.Logit(df['undefined_question_2'], sm.add_constant(df[[ 'bzsg_factor', 'redist_factor', 'undefined_question_1']]))
res1 = logit.fit()
print(res1.summary())

logit = sm.Logit(df['zsg_question_2'], sm.add_constant(df[[ 'bzsg_factor', 'redist_factor', 'zsg_question_1']]))
res2 = logit.fit()
print(res2.summary())

logit = sm.Logit(df['nzsg_question_2'], sm.add_constant(df[[ 'bzsg_factor', 'redist_factor', 'nzsg_question_1']]))
res3 = logit.fit()


textfile = open('statistics/output/regressions/logit_decide_to_redist_with_fairness.txt', 'w')
textfile.write(summary_col([res1, res2, res3],stars=True,float_format='%0.2f',model_names=['undefined\n(0)','zsg\n(1)','non-zsg\n(2)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),  'Pseudo R2':lambda x: "{:.2f}".format(x.prsquared)}).as_latex())
textfile.close()





#redistributed amount means
#Vignette: Fairness
mean = df[['undefined_question_1', 'zsg_question_1', 'nzsg_question_1']].mean()
std = df[['undefined_question_1', 'zsg_question_1', 'nzsg_question_1']].std()
error = std*ci_value/np.sqrt(len(df))
mean
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
for name, color in zip(vignettes, colors):
    mean = df[df[name+'_question_2'] == 1][name+'_question_3'].mean()
    b = df[df[name+'_question_2'] == 1][name+'_question_3'].plot.hist(bins=16, color=color, alpha=0.75)
    b.axvline(mean, color=color, linestyle='dashed', linewidth=1)
    b.set_title("Redistrubted Amount '" + name + "'")
    b.set_xlim(right=0.85)
    b.set_ylim(top=35)
    b.set_xlabel('Fraction Redistributed')
    b.get_figure().savefig('statistics/output/graphs/' + name + '_DistributionOfRedistAmount.png', dpi=resolution)
    plt.clf()


#change between question 3 and 4

df[df['undefined_question_3'] != df['undefined_question_4']]
diff_3_4  =df[df[name+'_question_2'] == 1][name+'_question_3'] - df[df[name+'_question_2'] == 1][name+'_question_3']
for name, color in zip(vignettes, colors):
    mean = df[df[name+'_question_2'] == 1][name+'_question_3'] .mean()
    mean1 = df[df[name+'_question_2'] == 1][name+'_question_4'] .mean()

    b = df[df[name+'_question_2'] == 1][name+'_question_3'].plot.hist(bins=16, color=color, alpha=0.75)
    b = df[df[name+'_question_2'] == 1][name+'_question_4'].plot.hist(bins=16, color=color, alpha=0.75)
    b.axvline(mean, color=color, linestyle='dashed', linewidth=1)
    b.axvline(mean1, color=color, linestyle='dotted', linewidth=1)
    b.set_title("Redistrubted Amount '" + name + "'")
    b.set_xlim(right=0.85)
    b.set_ylim(top=35)
    b.set_xlabel('Fraction Redistributed')
    b.get_figure().savefig('statistics/output/graphs/' + name + '_DistributionOfRedistAmount123.png', dpi=resolution)
    plt.clf()



#averages and correlations
fig, test = plt.subplots()
zsg_believers = df[df['bzsg_factor'] > df['bzsg_factor'].quantile(.90)] #50% people believers in zsg
zsg_non_believers = df[df['bzsg_factor'] < df['bzsg_factor'].quantile(.10)] #50% people believers in zsg
zsg_believers
for i in [1]:
    mean = zsg_believers['undefined_question_3'].mean()
    mean_non = zsg_non_believers['undefined_question_3'].mean()
    test = zsg_non_believers['undefined_question_3'].plot.hist(bins=16, color='red', alpha=0.75)
    test = zsg_believers['undefined_question_3'].plot.hist(bins=16, color='blue', alpha=0.75)
    test.axvline(mean, color='blue', linestyle='dashed', linewidth=1)
    test.axvline(mean_non, color='red', linestyle='dashed', linewidth=1)
    test.get_figure().savefig('statistics/output/graphs/' + 'believersvsnon.png', dpi=resolution)
    plt.clf()







# b.set_title("Redistrubted Amount '" + name + "'")
# b.set_xlim(right=0.85)
# b.set_ylim(top=35)
# b.set_xlabel('Fraction Redistributed')


#from scipy.stats import ks_2samp

#redist_factor on bzsg_factor graph
import seaborn as sns; sns.set(color_codes=True)

ax1 = sns.regplot(y='redist_factor', x='bzsg_factor', data=df)
ax1.set(xlabel='Belief in Zero-Sum', ylabel='Redistribution Preferences')
ax1.get_figure().savefig('statistics/output/graphs/redist_on_bzsg_factors.png', dpi=resolution)




df.corr(method='pearson')
