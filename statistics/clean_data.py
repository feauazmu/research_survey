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

#select uninteresting columns and drop them
drop_cols = [c for c in df.columns if c.lower()[:4] in ['part', 'sess', '1.gr', '1.su', '2.pa', '2.se']]
noninfo_cols = ['inequality_research_survey.1.player.id_in_group', 'inequality_research_survey.1.player.payoff',
    'inequality_research_survey.1.player.feedback', 'inequality_research_survey.1.subsession.round_number',
    'inequality_research_survey.1.group.id_in_subsession', 'inequality_research_survey.2.player.id_in_group',
    'inequality_research_survey.2.player.payoff', 'inequality_research_survey.2.player.feedback',
    'inequality_research_survey.2.subsession.round_number', 'inequality_research_survey.2.group.id_in_subsession',
    ]
drop_cols.extend(noninfo_cols)

df = df.drop(columns=drop_cols)
df = df[(pd.notnull(df['inequality_research_survey.1.player.gender']) & pd.notnull(df['inequality_research_survey.1.player.age']))]
df.columns

#bring answers in "2/1.player" to the same column. bszg and redistribution questions
for i in range(1,9):
    column_bzsg = 'inequality_research_survey.1.player.bzsg_{}'.format(i)
    column_bzsg2 = 'inequality_research_survey.2.player.bzsg_{}'.format(i)
    df[column_bzsg]= np.where(df[column_bzsg].isnull(), df[column_bzsg2], df[column_bzsg])

for i in range(1,5):
    column_redist = 'inequality_research_survey.1.player.redistribution_{}'.format(i)
    column_redist2 = 'inequality_research_survey.2.player.redistribution_{}'.format(i)
    df[column_redist]= np.where(df[column_redist].isnull(), df[column_redist2], df[column_redist])
df = df[pd.notnull(df['inequality_research_survey.1.player.bzsg_1'])]
df.reset_index()



#norm data and create indexes
df_bzsg = df.iloc[:,7:15]
#df_bzsg = df_bzsg[(pd.notnull(df_bzsg['inequality_research_survey.1.player.bzsg_1']))]


df_nbzsg = (df_bzsg-1)/6
df_nbzsg
df['nbzsg_avg'] = df_nbzsg.mean(axis=1, skipna = True)
df['nbzsg_avg']
df_redist = df.iloc[:,15:19]
    #invert question 4
#df_redist['inequality_research_survey.1.player.redistribution_4'] = 7 - df_redist['inequality_research_survey.1.player.redistribution_4']
df_redist = df_redist[(pd.notnull(df_redist['inequality_research_survey.1.player.redistribution_4']))]
df_nredist = (df_redist-1)/6
df['nredist_avg'] = df_nredist.mean(axis=1, skipna = True)




#Factor analysis
#Bartlett’s Test (is the true correlation matrix an identity matrix?)
chi_square_value,p_value=calculate_bartlett_sphericity(df_bzsg)
print(chi_square_value, p_value)
chi_square_value,p_value=calculate_bartlett_sphericity(df_nredist)
print(chi_square_value, p_value)

#Kaiser-Meyer-Olkin Test   (proportion of variance among all the observed variable) //value [0,1] > 0.6 => OK
kmo_all,kmo_model=calculate_kmo(df_nbzsg)    #not working with norm data! (determinant too close to zero)
print('bszg:', kmo_model)
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
pd.DataFrame(fa.loadings_)
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


# Get variance of each factors
pd.DataFrame(fa.get_factor_variance())
pd.DataFrame(fa.loadings_)
#create factor
df['redist_factor'] = -np.dot(df_nredist, pd.DataFrame(fa.loadings_))


#drop 2.player columns
drop_cols = [c for c in df.columns if c.lower()[:28] in ['inequality_research_survey.2']]
df = df.drop(columns=drop_cols)




#norm payments by treatment
for index, row in df.iterrows():
    if row['inequality_research_survey.1.player.treatment'] == ['treatment_1', 'treatment_2']:

        df['inequality_research_survey.1.player.undefined_question_3'] /= 800
        df['inequality_research_survey.1.player.undefined_question_4'] /= 800

        df['inequality_research_survey.1.player.zsg_question_3'] /= 50
        df['inequality_research_survey.1.player.zsg_question_4'] /= 50

        df['inequality_research_survey.1.player.nzsg_question_3'] /= 20
        df['inequality_research_survey.1.player.nzsg_question_4'] /= 20

    if row['inequality_research_survey.1.player.treatment'] == ['treatment_3', 'treatment_4']:

        df['inequality_research_survey.1.player.undefined_question_3'] /= 50
        df['inequality_research_survey.1.player.undefined_question_4'] /= 50

        df['inequality_research_survey.1.player.zsg_question_3'] /= 20
        df['inequality_research_survey.1.player.zsg_question_4'] /= 20

        df['inequality_research_survey.1.player.nzsg_question_3'] /= 800
        df['inequality_research_survey.1.player.nzsg_question_4'] /= 800

    if row['inequality_research_survey.1.player.treatment'] == ['treatment_5', 'treatment_6']:

        df['inequality_research_survey.1.player.undefined_question_3'] /= 20
        df['inequality_research_survey.1.player.undefined_question_4'] /= 20

        df['inequality_research_survey.1.player.zsg_question_3'] /= 800
        df['inequality_research_survey.1.player.zsg_question_4'] /= 800

        df['inequality_research_survey.1.player.nzsg_question_3'] /= 50
        df['inequality_research_survey.1.player.nzsg_question_4'] /= 50
df.columns

    if row['inequality_research_survey.1.player.treatment'] == ['treatment_1', 'treatment_2']:
        df['inequality_research_survey.1.player.undefined_question_3'] /= 800
        df['inequality_research_survey.1.player.undefined_question_4'] /= 800

df.to_csv(index=True, path_or_buf='cleaned_data.csv')
