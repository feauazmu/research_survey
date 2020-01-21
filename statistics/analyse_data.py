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
df.groupby('treatment').mean()[['undefined_question_3', 'zsg_question_3', 'nzsg_question_3']]
df[df['treatment']== 'treatment_6']['undefined_question_3']
#stata code
controls =  'age + C(gender) + C(education) + C(field_of_work) + C(nationality)+ C(treatment) + C(political_party)'
controls_as_list = []
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

#fairness on

mod = smf.ols(formula='undefined_question_1 ~ '+ controls  +   ' + redist_factor + bzsg_factor ', data=df)

res = mod.fit()
print(res.summary())
mod = smf.ols(formula='undefined_question_1 ~ bzsg_factor + redist_factor', data=df)

res1 = mod.fit()
print(res.summary())

mod = smf.ols(formula='undefined_question_1 ~ bzsg_factor ', data=df)

res2 = mod.fit()
print(res.summary())

mod = smf.ols(formula='undefined_question_1 ~ redist_factor' , data=df)

res3 = mod.fit()
print(res.summary())

textfile = open('statistics/output/regressions/undefined_fairness_on_factors.txt', 'w')
textfile.write(summary_col([res2, res3, res1, res],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)','\n(2)','\n(3)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)}).as_latex())
print(summary_col([res2, res3, res1, res],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)','\n(2)','\n(3)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)}).as_latex())
textfile.close()


#redist amount on interaction factor
df['redistXfair'] = df['undefined_question_1']*df['redist_factor']
mod = smf.ols(formula='undefined_question_3 ~ '+ ' + redist_factor ', data=df)

res = mod.fit()
print(res.summary())
mod = smf.ols(formula='undefined_question_3 ~ undefined_question_1', data=df)

res1 = mod.fit()
print(res.summary())

mod = smf.ols(formula='undefined_question_3 ~ undefined_question_1 + redist_factor  ', data=df)

res2 = mod.fit()
print(res.summary())

mod = smf.ols(formula='undefined_question_3 ~ undefined_question_1 + redist_factor + redistXfair'  , data=df)

res3 = mod.fit()
print(res.summary())

mod = smf.ols(formula='undefined_question_3 ~ redistXfair '  , data=df)

res4 = mod.fit()
print(res.summary())

textfile = open('statistics/output/regressions/undefined_redist_amount_on_interaction_factors.txt', 'w')
textfile.write(summary_col([res, res1, res2, res3,res4],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)','\n(2)','\n(3)','\n(4)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)}).as_latex())
print(summary_col([res, res1, res2, res3,res4],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)','\n(2)','\n(3)', '\n(4)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),'R2':lambda x: "{:.2f}".format(x.rsquared)}).as_latex())
textfile.close()

df.columns
#logit prob of redistribute
for vign in vignettes:
    logit = smf.logit(formula=vign + '_question_2 ~ bzsg_factor ', data=df)
    res1 = logit.fit()
    print(res1.summary())

    logit = smf.logit(formula=vign + '_question_2 ~ bzsg_factor + redist_factor', data=df)
    res2 = logit.fit()
    print(res2.summary())

    logit = smf.logit(formula=vign + '_question_2 ~ bzsg_factor + redist_factor + ' + vign + '_question_1', data=df)
    res3 = logit.fit()

    logit = smf.logit(formula=vign + '_question_2 ~ bzsg_factor + redist_factor + ' + vign + '_question_1 + age + C(political_party) + C(gender) + C(field_of_work) + C(nationality)' , data=df)
    res4 = logit.fit()

    textfile = open('statistics/output/regressions/'+ vign +'_logit_decide_to_redist.txt', 'w')
    textfile.write(summary_col([res1, res2, res3, res4],stars=True,float_format='%0.2f',model_names=['\n(0)','\n(1)','\n(2)','\n(3)'], info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),  'Pseudo R2':lambda x: "{:.2f}".format(x.prsquared)}).as_latex())
    textfile.close()

df.groupby('education').count()




#redistributed amount means
#Vignette: Fairness
mean = df[['undefined_question_1', 'zsg_question_1', 'nzsg_question_1']].mean()
mean
std = df[['undefined_question_1', 'zsg_question_1', 'nzsg_question_1']].std()
error = std*ci_value/np.sqrt(len(df))
fig, mra = plt.subplots()
mra.bar(vignette_names, mean, color=colors, yerr=error, align='center', alpha=0.75, ecolor='black', capsize=10)

mra.set_title('Vignettes: Fairness')
mra.set_ylabel('How unfair')
mra.set_xticklabels(vignette_names, rotation=0)
plt.figtext(0.2, 0.01, '*95%-Confidence intervals', horizontalalignment='right', fontdict={'size': 'xx-small'})

mra.get_figure().savefig('statistics/output/graphs/Vignettes_Fairness.png', dpi=resolution)

plt.clf()

#Vignttes want to redistribute
mean = df[['undefined_question_2', 'zsg_question_2', 'nzsg_question_2']].mean()
std = df[['undefined_question_2', 'zsg_question_2', 'nzsg_question_2']].std()
error = std*ci_value/np.sqrt(len(df))

fig, mra = plt.subplots()
mra.bar(vignette_names, mean, color=colors, yerr=error, align='center', alpha=0.75, ecolor='black', capsize=10)

mra.set_title('Vignettes: Redistribute')
mra.set_ylabel('Fraction')
mra.set_xticklabels(vignette_names, rotation=0)
plt.figtext(0.2, 0.01, '*95%-Confidence intervals', horizontalalignment='right', fontdict={'size': 'xx-small'})

mra.get_figure().savefig('statistics/output/graphs/Vignettes_Redistribute.png', dpi=resolution)
plt.clf()

#Vignettes mean redistributed amount
mean = df[['undefined_question_3', 'zsg_question_3', 'nzsg_question_3']].mean()
std = df[['undefined_question_3', 'zsg_question_3', 'nzsg_question_3']].std()
error = std*ci_value/np.sqrt(len(df))

fig, mra = plt.subplots()
mra.bar(vignette_names, mean, color=colors, yerr=error, align='center', alpha=0.75, ecolor='black', capsize=10)

mra.set_title('Vignettes: Mean Redistributed Amount')
mra.set_ylabel('Fraction')
mra.set_xticklabels(vignette_names, rotation=0)
plt.figtext(0.2, 0.01, '*95%-Confidence intervals', horizontalalignment='right', fontdict={'size': 'xx-small'})

mra.get_figure().savefig('statistics/output/graphs/Vignettes_Mean_Redistributed_Amount.png', dpi=resolution)
plt.clf()

#Vignettes Updated redistributed amount
mean = df[['undefined_question_4', 'zsg_question_4', 'nzsg_question_4']].mean()
std = df[['undefined_question_4', 'zsg_question_4', 'nzsg_question_4']].std()
error = std*ci_value/np.sqrt(len(df))

fig, mra = plt.subplots()
mra.bar(vignette_names, mean, color=colors, yerr=error, align='center', alpha=0.75, ecolor='black', capsize=10)

mra.set_title('Vignettes: Updated Mean Redistributed Amount')
mra.set_ylabel('Fraction')
mra.set_xticklabels(vignette_names, rotation=0)
plt.figtext(0.2, 0.01, '*95%-Confidence intervals', horizontalalignment='right', fontdict={'size': 'xx-small'})

mra.get_figure().savefig('statistics/output/graphs/Vignettes_Updated_Mean_Redistributed_Amount.png', dpi=resolution)

plt.clf()
#Distribution of reddistributed amount
for name, color in zip(vignettes, colors):
    mean = df[df[name+'_question_2'] == 1][name+'_question_3'].mean()
    b = df[df[name+'_question_2'] == 1][name+'_question_3'].plot.hist(bins=16, color=color, alpha=0.75)
    b.axvline(mean, color=color, linestyle='dashed', linewidth=1)
    b.set_title("Redistributed Amount '" + name + "'")
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
    b.set_title("Redistibuted Amount '" + name + "'")
    b.set_xlim(right=0.85)
    b.set_ylim(top=35)
    b.set_xlabel('Fraction Redistributed')
    b.get_figure().savefig('statistics/output/graphs/' + name + '_DistributionOfRedistAmount123.png', dpi=resolution)
    plt.clf()



#averages and correlations
fig, test = plt.subplots()
zsg_believers = df[df['bzsg_factor'] > df['bzsg_factor'].quantile(.75)] #50% people believers in zsg
zsg_non_believers = df[df['bzsg_factor'] < df['bzsg_factor'].quantile(.25)] #50% people believers in zsg
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

#redist_diff graphs on effort_diff
df[vign + '_redist_diff'] = 0
for vign in vignettes:
    df[vign + '_redist_diff'] = (df[vign + '_question_4'] - df[vign + '_question_3'])
a = df[['undefined_question_3', 'undefined_question_4', 'undefined_redist_diff']]
a[a['undefined_redist_diff'] < 0].count()
df = df.sort_values(by=['undefined_effort_diff'])
mean = df.groupby('undefined_effort_diff')
df[df['undefined_effort_diff_d'] == 1]['undefined_redist_diff']
mean.plot.bar(y='nzsg_redist_diff', color=color, alpha=0.75)
df[df['undefined_question_2'] != 0]['undefined_redist_diff'].plot.hist()
df['nzsg_effort_diff'].max()
df['undefined_redist_diff']
plt.clf()
i = 1
means = []
errors = []
for vign, color in zip(vignettes, colors):

    mean = (df.groupby(vign + '_effort_diff').mean())[vign + '_redist_diff']
    std = df.groupby(vign + '_effort_diff').std()[vign + '_redist_diff']
    error = std*ci_value/np.sqrt(len(df))
    means.append(mean)
    errors.append(error)
    b = mean.plot.bar( y=vign+'_redist_diff', color=color, alpha=0.75, yerr=error, align='center',ecolor='black', capsize=10,)

    b.set_title('Update on Effort Change')
    b.set_ylabel('Change in Redistributed Amount [p.p.]')
    b.set_xlabel('Change in Effort Difference [p.p.]')
    b.set_xticklabels([9, 33, 38],rotation=0)
    plt.figtext(0.2, 0.01, '*95%-Confidence intervals', horizontalalignment='right', fontdict={'size': 'xx-small'})

    b.get_figure().savefig('statistics/output/graphs/' + vign + '_redist_vs_effort_diff.png', dpi=resolution)
    i +1
    plt.clf()





#create redist vs effor diff summary
for i in [1]:
    df1 = pd.DataFrame(means).T*100
    df2 = pd.DataFrame(errors).T*100
    df2
    df1
    b = df1.plot.bar(color=colors, alpha=0.75, yerr=df2, align='center',ecolor='black', capsize=10)
    b.set_title('Update on Effort Change')
    b.set_ylabel('Change in Redistributed Amount [p.p.]')
    b.set_xlabel('Change in Effort Difference [p.p.]')
    b.set_xticklabels([9, 33, 38],rotation=0)

    plt.figtext(0.01, 0.01, '*95%-Confidence intervals', fontdict={'size': 'xx-small'})
    b.axvline(1.5, color='black', linestyle='dashed', linewidth=0.8)
    handles, labels = b.get_legend_handles_labels()
    handles
    b.legend(handles, vignette_names, loc="lower left")

    plt.figtext(.67, .85, 'Winner works harder', fontdict={'size': 'small'})
    plt.figtext(.4, .85, 'Loser works less', fontdict={'size': 'small'})

    b.get_figure().savefig('statistics/output/graphs/redist_vs_effort_diff_summary.png', dpi=resolution)

plt.clf()


#redist_factor on bzsg_factor graph
import seaborn as sns; sns.set(color_codes=True)

ax1 = sns.regplot(y='redist_factor', x='bzsg_factor', data=df)
ax1.set(xlabel='Belief in Zero-Sum', ylabel='Redistribution Preferences')
ax1.get_figure().savefig('statistics/output/graphs/redist_on_bzsg_factors.png', dpi=resolution)



(df.mean()['nzsg_question_3']-df.mean()['zsg_question_3'])/df.mean()['zsg_question_3']


avg = (df.mean()['nzsg_question_3']+df.mean()['zsg_question_3'])/2


(df.mean()['undefined_question_3'] - avg)/avg
(df.mean()['undefined_question_3']-df.mean()['nzsg_question_3'])/df.mean()['nzsg_question_3']
