import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo


df = pd.read_csv('statistics/inequality_research_survey.csv')

vignettes = ['undefined', 'zsg', 'nzsg']

#select uninteresting columns to drop them
drop_cols = [c for c in df.columns if c.lower()[:4] in ['part', 'sess', '1.gr', '1.su', '2.pa', '2.se']]
noninfo_cols = ['index', 'inequality_research_survey.1.player.id_in_group', 'inequality_research_survey.1.player.payoff',
    'inequality_research_survey.1.player.feedback', 'inequality_research_survey.1.subsession.round_number',
    'inequality_research_survey.1.group.id_in_subsession', 'inequality_research_survey.2.player.id_in_group',
    'inequality_research_survey.2.player.payoff', 'inequality_research_survey.2.player.feedback',
    'inequality_research_survey.2.subsession.round_number', 'inequality_research_survey.2.group.id_in_subsession',
    ]

player2_cols = [c for c in df.columns if c.lower()[:28] in ['inequality_research_survey.2']]

drop_cols.extend(noninfo_cols)
drop_cols.extend(player2_cols)



#bring answers in "2/1.player" to the same column. bzsg and redistribution questions
for i in range(1,9):
    column_bzsg = 'inequality_research_survey.1.player.bzsg_{}'.format(i)
    column_bzsg2 = 'inequality_research_survey.2.player.bzsg_{}'.format(i)
    df[column_bzsg]= np.where(df[column_bzsg].isnull(), df[column_bzsg2], df[column_bzsg])

for i in range(1,5):
    column_redist = 'inequality_research_survey.1.player.redistribution_{}'.format(i)
    column_redist2 = 'inequality_research_survey.2.player.redistribution_{}'.format(i)
    df[column_redist]= np.where(df[column_redist].isnull(), df[column_redist2], df[column_redist])


df = df.reset_index()

#drop columns
df = df.drop(columns=drop_cols)

#rename columns
short_columns = [c.replace('inequality_research_survey.1.player.', '') for c in df.columns]
df.columns = short_columns

#drop incomplete rows
df =df.dropna()

#norm data and create indexes
df_bzsg = df.iloc[:,7:15]

#df_bzsg = df_bzsg[(pd.notnull(df_bzsg['bzsg_1']))]
df_nbzsg = (df_bzsg-1)/6

df['nbzsg_avg'] = df_nbzsg.mean(axis=1, skipna = True)

df_redist = df.iloc[:,15:19]
    #invert question 4
df_redist['redistribution_4'] = 8 - df_redist['redistribution_4']
#df_redist = df_redist[(pd.notnull(df_redist['redistribution_4']))]
df_nredist = (df_redist-1)/6
df_nredist
df['nredist_avg'] = df_nredist.mean(axis=1, skipna = True)
df['nredist_avg'].max()
#Factor analysis
#Bartlett’s Test (is the true correlation matrix an identity matrix?)
chi_square_value,p_value=calculate_bartlett_sphericity(df_nbzsg)
print(chi_square_value, p_value)
chi_square_value,p_value=calculate_bartlett_sphericity(df_nredist)
print(chi_square_value, p_value)

#Kaiser-Meyer-Olkin Test   (proportion of variance among all the observed variable) //value [0,1] > 0.6 => OK
kmo_all,kmo_model=calculate_kmo(df_nbzsg)
print('bzsg:', kmo_model)
kmo_all,kmo_model=calculate_kmo(df_nredist)
print('redist:', kmo_model)

#apply
#bzsg
# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(n_factors = 1, rotation=None)
fa.fit(df_nbzsg)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

# Create scree plot using matplotlib
plt.scatter(range(1,df_nbzsg.shape[1]+1),ev)
plt.plot(range(1,df_nbzsg.shape[1]+1),ev)
#commands for jupiter notebook
# plt.title('Scree Plot')
# plt.xlabel('Factors')
# plt.ylabel('Eigenvalue')
# plt.grid()
# plt.show()


# Get variance of each factors
pd.DataFrame(fa.get_factor_variance())
b=pd.DataFrame(fa.loadings_)
b.sum()
#create factor
df['bzsg_factor'] = np.dot(df_nbzsg, pd.DataFrame(fa.loadings_))

#redist
fa = FactorAnalyzer(n_factors = 1, rotation=None)
fa.fit(df_nredist)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

# Create scree plot using matplotlib
plt.scatter(range(1,df_nredist.shape[1]+1),ev)
plt.plot(range(1,df_nredist.shape[1]+1),ev)
#commands for jupiter notebook
# plt.title('Scree Plot')
# plt.xlabel('Factors')
# plt.ylabel('Eigenvalue')
# plt.grid()
# plt.show()
pd.DataFrame(fa.get_factor_variance())
a = pd.DataFrame(fa.loadings_)
a.sum()
# Get variance of each factors
#pd.DataFrame(fa.get_factor_variance())
#pd.DataFrame(fa.loadings_)
#create factor
df['redist_factor'] = -np.dot(df_nredist, pd.DataFrame(fa.loadings_))

#norm payments by treatment
df['undefined_effort_diff'] = 0.0
df['zsg_effort_diff'] = 0.0
df['nzsg_effort_diff'] = 0.0
df['undefined_effort_diff_d'] = 0
df['zsg_effort_diff_d'] = 0
df['nzsg_effort_diff_d'] = 0

df['undefined_effort_diff'].max()
df['undefined_question_3'].max()

df[df['nzsg_question_3'] < 5]['nzsg_question_3'].mean()

for index, row in df.iterrows():
    if row['treatment'] in ['treatment_1', 'treatment_2']:

        df.at[index, 'undefined_question_3'] = row['undefined_question_3'] / 800
        df.at[index, 'undefined_question_4'] = row['undefined_question_4'] / 800
        df.at[index, 'undefined_effort_diff'] = 0.333
        df.at[index, 'undefined_effort_diff_d'] = 0

        df.at[index, 'zsg_question_3'] = row['zsg_question_3'] / 50
        df.at[index, 'zsg_question_4'] = row['zsg_question_4'] / 50
        df.at[index, 'zsg_effort_diff'] = 0.099
        df.at[index, 'zsg_effort_diff_d'] = 0

        if df.at[index, 'nzsg_question_2'] == 1:
            if df.at[index, 'nzsg_question_3'] < .26*20*0:
                df.at[index, 'nzsg_question_3'] = row['nzsg_question_3'] / 20
                df.at[index, 'nzsg_question_4'] = row['nzsg_question_4'] / 20
            else:
                df.at[index, 'nzsg_question_3'] = 1 - row['nzsg_question_3'] / 20
                df.at[index, 'nzsg_question_4'] = 1 - row['nzsg_question_4'] / 20
        df.at[index, 'nzsg_effort_diff'] = 0.385
        df.at[index, 'nzsg_effort_diff_d'] = 1

    if row['treatment'] in ['treatment_3', 'treatment_4']:

        df.at[index, 'undefined_question_3'] = row['undefined_question_3'] / 50
        df.at[index, 'undefined_question_4'] = row['undefined_question_4'] / 50
        df.at[index, 'undefined_effort_diff'] = .099
        df.at[index, 'undefined_effort_diff_d'] = 0

        if df.at[index, 'zsg_question_2'] == 1:
            if df.at[index, 'zsg_question_3'] < .44*20*0:
                df.at[index, 'zsg_question_3'] = row['zsg_question_3'] / 20
                df.at[index, 'zsg_question_4'] = row['zsg_question_4'] / 20
            else:
                df.at[index, 'zsg_question_3'] = 1 - row['zsg_question_3'] / 20
                df.at[index, 'zsg_question_4'] = 1 - row['zsg_question_4'] / 20
        df.at[index, 'zsg_effort_diff'] = .385
        df.at[index, 'zsg_effort_diff_d'] = 1

        df.at[index, 'nzsg_question_3'] = row['nzsg_question_3'] / 800
        df.at[index, 'nzsg_question_4'] = row['nzsg_question_4'] / 800
        df.at[index, 'nzsg_effort_diff'] = .333
        df.at[index, 'nzsg_effort_diff_d'] = 0

    if row['treatment'] in ['treatment_5', 'treatment_6']:

        if df.at[index, 'undefined_question_2'] == 1:
            if df.at[index, 'undefined_question_3'] < .26*20*0:
                df.at[index, 'undefined_question_3'] = row['undefined_question_3'] / 20
                df.at[index, 'undefined_question_4'] = row['undefined_question_4'] / 20
            else:
                df.at[index, 'undefined_question_3'] = 1 - row['undefined_question_3'] / 20
                df.at[index, 'undefined_question_4'] = 1 - row['undefined_question_4'] / 20
        df.at[index, 'undefined_effort_diff'] = 0.385
        df.at[index, 'undefined_effort_diff_d'] = 1

        df.at[index, 'zsg_question_3'] = row['zsg_question_3'] / 800
        df.at[index, 'zsg_question_4'] = row['zsg_question_4'] / 800
        df.at[index, 'zsg_effort_diff'] = .333
        df.at[index, 'zsg_effort_diff_d'] = 0

        df.at[index, 'nzsg_question_3'] = row['nzsg_question_3'] / 50
        df.at[index, 'nzsg_question_4'] = row['nzsg_question_4'] / 50
        df.at[index, 'nzsg_effort_diff'] = .099
        df.at[index, 'nzsg_effort_diff_d'] = 0

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

#drop_cols.append('Unnamed: 0')
drop_cols.append('zero_redist1')
df = df.drop(columns=drop_cols)

#normalize factors
df['bzsg_factor']= -df['bzsg_factor']/3.887762
df['redist_factor'] = df['redist_factor']/1.871011

#normalize question 1 fairness
for vign in vignettes:
    df[vign + '_question_1'] = (df[vign + '_question_1'] - 1)/6

df.to_csv(index=True, path_or_buf='cleaned_data.csv')

for vign in vignettes:
    print(vign)
    corrected_data = df[df[vign + '_effort_diff_d'] == 1]
    right_data = df[df[vign + '_effort_diff_d'] == 0]

    right_data[vign + '_redist_diff'] = right_data[vign + '_question_3']-right_data[vign + '_question_4']
    print(right_data[right_data[vign + '_redist_diff'] < 0][[vign + '_question_3', vign + '_question_4', vign + '_redist_diff']])

    corrected_data[vign + '_redist_diff'] = corrected_data[vign + '_question_3']-corrected_data[vign + '_question_4']
    print(corrected_data[corrected_data[vign + '_redist_diff'] < 0][[vign + '_question_3', vign + '_question_4', vign + '_redist_diff']])


    diff_corrected = corrected_data[vign + '_question_3'] - corrected_data[vign + '_question_4']
    diff_right = right_data[vign + '_question_3'] - right_data[vign + '_question_4']
    (diff_corrected != 0).sum()
    (diff_right != 0).sum()

    print('right data dif > 0: ' + str((diff_right >0).sum()
    ))
    print('right data dif < 0: ' + str((diff_right <0).sum()
    ))
    print('corrected data dif > 0: ' + str((diff_corrected >0).sum()))
    print('corrected data dif < 0: ' + str((diff_corrected <0).sum()))

    print('right data mean diff redist. amount: '+ str(diff_right.mean()))
    print('corrected data mean diff redist. amount: '+ str(diff_corrected.mean()))
    print('diff in diff: ' + str(diff_right.mean() - diff_corrected.mean()))

print()
#test corrected data
from scipy.stats import ks_2samp
plt.clf()
for vign in vignettes:
    corrected_data = df[df[vign + '_effort_diff_d'] == 1]
    right_data = df[df[vign + '_effort_diff_d'] == 0]

    corrected_data = corrected_data[corrected_data[vign + '_question_2'] == 1][vign + '_question_3']
    right_data = right_data[right_data[vign + '_question_2'] == 1][vign + '_question_3']
    A = right_data.plot.hist(bins=16, color='blue', alpha=0.50, weights=np.zeros_like(right_data) + 1. / right_data.size)
    A = corrected_data.plot.hist(bins=16, color='red', alpha=0.50, weights=np.zeros_like(corrected_data) + 1. / corrected_data.size)
    A.get_figure().savefig('statistics/output/correction/' + vign + '_correctedvsright_distribution.png')

    print(vign)
    print('corrected N: ' + str(len(corrected_data)))
    print('right N ' + str(len(right_data)))
    print(ks_2samp(corrected_data, right_data))
    plt.clf()

dict = {'undefined': [], 'zsg': [], 'nzsg': []}
dict['undefined']
dict

#benchmark to test 2 random samples of right data
for i in range(100):
    for vign in vignettes:
        corrected_data = df[df[vign + '_effort_diff_d'] == 1]
        right_data = df[df[vign + '_effort_diff_d'] == 0]

        corrected_data = corrected_data[corrected_data[vign + '_question_2'] == 1][vign + '_question_3']
        right_data = right_data[right_data[vign + '_question_2'] == 1][vign + '_question_3']
        # A = right_data.plot.hist(bins=32, color='blue', alpha=0.50, weights=np.zeros_like(right_data) + 1. / right_data.size)
        # A = corrected_data.plot.hist(bins=32, color='red', alpha=0.50, weights=np.zeros_like(corrected_data) + 1. / corrected_data.size)
        # A.get_figure().savefig('statistics/output/correction/' + vign + '_correctedvsright_distribution.png')
        msk = np.random.rand(len(right_data)) < 0.5
        part1 = right_data[msk]
        part2 = right_data[~msk]
        print(vign)
        print('corrected N: ' + str(len(corrected_data)))
        print('right N ' + str(len(right_data)))
        dict[vign].append(ks_2samp(part1, part2).pvalue)
        print(ks_2samp(part1, part2))

dict

pd.DataFrame.from_dict(dict).mean()

#find max redist amount for right data
for vign in vignettes:
    corrected_data = df[df[vign + '_effort_diff_d'] == 1]
    right_data = df[df[vign + '_effort_diff_d'] == 0]
    print(vign + str(right_data[vign + '_question_3'].max()))
    # outliers = corrected_data[corrected_data[vign + 'zsg_question_3'] < right_data[vign + '_question_3'].max()]
    # outliers[] = outliers
