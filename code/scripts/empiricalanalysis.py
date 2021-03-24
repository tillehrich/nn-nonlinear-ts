import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import bds
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from arch.univariate import arch_model
from pmdarima.arima import auto_arima

import re
import warnings
from tqdm import tqdm
from  joblib import Parallel, delayed

plt.rcParams['figure.figsize'] = [12, 8]

def FitGARCH(series, p, q):
    """
    :param series: series to fit on
    :param p: lag length of residuals
    :param q: lag length of variances
    :return: dictionary of model configuration and result of arch-lm test
    """
    am = arch_model(series, mean='Zero', vol='GARCH', p=p, q=q, rescale=False).fit(disp='off')

    # calculate standardized residuals
    series_std = series / np.sqrt(am.conditional_volatility)

    # append result of arch lm test with degrees of freedom of p+q to output
    archresult = het_arch(series_std, nlags=20, ddof=p + q)[1]
    return [(p,q), archresult]

if __name__ == '__main__':

    ################################################################################
    # CONTROL PANEL
    ################################################################################
    multiproc = True
    ################################################################################

    ### Data import and preparation
    # Read in Data
    # Downloaded from https://research.stlouisfed.org/econ/mccracken/fred-databases/ on 22.12.2020
    df = pd.read_csv('../data/current.csv')

    #drop rows that contain exclusively null values
    df = df.dropna(how='all')

    #drop first row, as first row contains info on transformations, not real data
    df = df.drop(0)

    #define date column
    df.sasdate = pd.to_datetime(df['sasdate'])

    #set date column as index
    df = df.set_index('sasdate')

    series_name = 'HOUSTMW'

    # define series to use
    series = df[series_name]

    #series plot
    plt.plot(series)
    plt.savefig('../../manuscript/src/latexGraphics/houstmw.png')
    #plt.show()
    plt.close()

    #create dictionary for command creation in LaTex, which is later stored in Variables.tex file
    dct_tex = {}

    ### ADF Test
    # p-value of ADF Test, regression='c' stands for constant
    #https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
    dct_tex['adfresult'] = np.round(adfuller(series, regression = 'c')[1], 4)


    #plot the autocorrelation function of the series
    #https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html
    plot_acf(series, lags  =60, title = None)
    plt.savefig('../../manuscript/src/latexGraphics/acf_houstmw.png')
    #plt.show()
    plt.close()

    #plot the partial autocorrelation function of the series
    #https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_pacf.html
    plot_pacf(series, lags  =60, title = None)
    plt.savefig('../../manuscript/src/latexGraphics/pacf_houstmw.png')
    #plt.show()
    plt.close()

    #fit optimal ARIMA model
    #https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
    arima_model = auto_arima(series, start_p=0, d=0, start_q=0, max_p=40, max_d=0, max_q=40)

    #save results
    dct_tex['parima'] = arima_model.order[0]
    dct_tex['qarima'] = arima_model.order[2]

    warnings.filterwarnings("ignore")
    model1 = ARIMA(series, order=(arima_model.order[0], 0, arima_model.order[2]))
    warnings.filterwarnings("default")
    model1_fit = model1.fit()

    #write model output to latex tables for appendix
    str_out = ''
    for i, table in enumerate(model1_fit.summary().tables):
        caption = 'ARIMA Results ' + str(i+1)
        df_str=pd.read_html(table.as_html(), header=0, index_col=0)[0].fillna('').to_latex(caption = caption)+'\n'

        str_tmp = re.split(r'(aption{' + caption + r'})', df_str)

        str_out += str_tmp[0] + str_tmp[1] + '\n' + r'\label{tab:' + 'arima'+str(i+1) + r'}' + str_tmp[2]

    with open(
            f'../../manuscript/src/latexTables/ArimaResults.tex', 'w'
    ) as tf:
        tf.write(str_out)

    #obtain fitted values
    y_hat_arima = model1_fit.predict()

    #obtain residuals
    u_hat = series-y_hat_arima

    #generate plot of residuals
    plt.plot(u_hat)
    plt.savefig('../../manuscript/src/latexGraphics/u_hat_houstmw.png')
    #plt.show()
    plt.close()

    #generate acf plot of squared residuals
    plot_acf(u_hat**2, lags =100)
    plt.savefig('../../manuscript/src/latexGraphics/acf_u_hat_sq_houstmw.png')
    plt.close()

    #arch-lm test
    #https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html
    dct_tex['archlmresult']=np.round(het_arch(u_hat, nlags=3,ddof = 6)[1], 4)

    ####select optimal GARCH Model according to arch lm test result of standardized residuals

    #define grid of model parameters:
    ls_params = [(p, q) for p in range(1, 20) for q in range(1, 20)]
    ls_out = []
    if multiproc == True:

        ls_out = Parallel(n_jobs = -1)(delayed(FitGARCH)(*[u_hat, param[0], param[1]]) for param in ls_params)

    else:
        for param in tqdm(ls_params):
            ls_out.append(FitGARCH(u_hat, param[0], param[1]))


    #convert output to table
    df = pd.DataFrame(ls_out, columns = ['order', 'archlmresult'])

    #select optimal garch order according to maximum p-value
    garch_order = df.iloc[df.archlmresult.idxmax(), :].order

    #save results
    dct_tex['pgarch'] = garch_order[0]
    dct_tex['qgarch'] = garch_order[1]
    dct_tex['archlmresultstd'] = np.round(df[df.order == garch_order]['archlmresult'].values[0], 4)

    #obtain final residual series by refitting the model with best parameter combination:
    am = arch_model(u_hat, mean='Zero', vol='GARCH', p=garch_order[0], q=garch_order[1], rescale=False).fit(
        disp = 'off')

    u_hat_std = u_hat / np.sqrt(am.conditional_volatility)

    #BDS Test, epsilon is chosen according to

    epsilon = 0.5*np.std(u_hat_std)
    max_dim = 5
    bdsstat = bds(np.log(u_hat_std ** 2), max_dim = max_dim, epsilon = epsilon)[1]

    df = pd.DataFrame()
    df['m'] = [str(i) for i in range(2, max_dim + 1)]
    df['p-value'] = bdsstat

    caption = 'BDS Test Results'

    df_str = df.T.to_latex(caption=caption,header = False)

    #split latex string and insert reference
    str_tmp = re.split(r'(aption{'+caption+r'})', df_str)

    str_out = str_tmp[0] + str_tmp[1] + '\n' + r'\label{tab:' + 'bds' + r'}' + str_tmp[2]

    with open(
            '../../manuscript/src/latexTables/BDSResults.tex', 'w'
    ) as tf:
        tf.write(str_out)

    #save variables to Variables.tex
    str_out = r''
    for i in dct_tex:
        str_out+= r"\newcommand"'\\'+i+r'{'+str(dct_tex[i])+r'}'+'\n'

    with open(
        '../../manuscript/src/latexChapters/Variables.tex', 'w'
    ) as tf:
        tf.write(str_out)
