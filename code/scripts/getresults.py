import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [6, 4]

#define paths to output of benchmark script

dct_files = {
    '1': 'results_series_name_HOUSTMW_n_ahead_1_earliest_endpoint_692_9759d6c06e8549a49ce9f5aa1c3adc95.pkl',
    '5': 'results_series_name_HOUSTMW_n_ahead_5_earliest_endpoint_692_fd8076f16af74ee3b3537116be9cbd0e.pkl',
    '10': 'results_series_name_HOUSTMW_n_ahead_10_earliest_endpoint_692_c0c1e1fc4c224199bc5127b6b62e505c.pkl'
}

ls_out = []
dct_str = {}
str_final = ''
for n_ahead, filename in dct_files.items():

    filepath = '../results/'+filename
    #open dataframe with results for each subsample
    filehandler = open(filepath, 'rb')

    df_res = pickle.load(filehandler)

    plt.plot(df_res['MSE_MLP'], label = 'MLP')
    plt.plot(df_res['MSE_ARMA(p,q)'], label = 'ARMA')
    plt.legend()
    plt.title(f'horizon = {n_ahead}')
    plt.savefig(f'../../manuscript/src/latexGraphics/benchmark_{n_ahead}_ahead.png')
    plt.close()

    ls_out.append(df_res[['MSE_MLP', 'MSE_ARMA(p,q)']].mean())

    #creating output tables
    df_res = df_res[['endpoint', 'MSE_MLP', 'sizes', 'alpha', 'MSE_ARMA(p,q)', 'p', 'q']]

    df_res.columns = ['Endpoint', 'MSE(MLP)', 'Sizes', 'Lambda', 'MSE(ARMA)', 'p', 'q']

    df_res = df_res[:20]

    df_res = np.round(df_res, 4)

    caption = f'Model specifications for horizon {n_ahead} for first 20 endpoints'

    df_str = df_res.to_latex(caption = caption, index = False)

    # split latex string and insert reference
    str_tmp = re.split(r'(aption{' + caption + r'})', df_str)
    str_out = str_tmp[0] + str_tmp[1] + '\n' + r'\label{tab:' + f'specifications{n_ahead}' + r'}' + str_tmp[2]

    str_final += str_out+'\n'


#Output LaTex tables
#Specifications

with open(
        '../../manuscript/src/latexTables/Specifications.tex', 'w'
) as tf:
    tf.write(str_final)


#MSE
df_out = pd.DataFrame(ls_out)
df_out['h'] = dct_files.keys()
df_out = df_out[['h', 'MSE_MLP', 'MSE_ARMA(p,q)']]
df_out.columns = ['h', 'MLP', 'ARMA']

df_out = np.round(df_out, 2)

caption = 'Mean MSE for different forecast horizons'

df_str = df_out.to_latex(caption = caption, index = False)

# split latex string and insert reference
str_tmp = re.split(r'(aption{' + caption + r'})', df_str)

str_out = str_tmp[0] + str_tmp[1] + '\n' + r'\label{tab:' + 'mse' + r'}' + str_tmp[2]

with open(
        '../../manuscript/src/latexTables/MSE.tex', 'w'
) as tf:
    tf.write(str_out)
