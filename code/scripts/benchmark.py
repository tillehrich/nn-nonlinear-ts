import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

import pickle
from  joblib import Parallel, delayed
import time
import warnings
import uuid
from tqdm import tqdm
import itertools
import random


def NodesizeCombinations(max_nodesize, max_depth, n_models=None, stepsize_nodes = 1):
    """
    :param max_nodesize: maximum nodesize
    :param max_depth: maximum number of layers for mlp
    :param n_models: maximum number of layers for mlp
    :return: either list of all ordered combinations with replacement of all lengths up to max_depth or random subset of the latter
    """
    #case if there is no random sampling
    #for every possible length, sample all ordered combinations of all possible nodesizes and add to ls_out(nested list of tuples at this point)
    if n_models == None:
        ls_out = []
        for depth in range(1, max_depth + 1):
            ls_out.append(list(itertools.product(range(1, max_nodesize + 1, stepsize_nodes), repeat=depth)))
        #flatten ls_out into plain list of tuples
        ls_out = list(itertools.chain(*ls_out))
    #case if there is random sampling
    else:
        ls_out = []
        #for each depth, sample nodesizes without replacement in order and append tuples to output list number of samples is chosen to be as low as possible (memory issues) but definitely
        #larger than n_models
        for depth in range(1, max_depth + 1):
            for i in range(int(n_models/max_depth)):
                ls_out.append(tuple(random.choices(range(1, max_nodesize + 1), k=depth)))
    return ls_out

def CreateTrainFrame(series, lags, sasdate_index):
    """
    :param series: series to create training frame with
    :param lags: number of lags to include
    :return: dataframe with original series and column for each lag. Rows with null values are dropped
    """

    #initialize empty dataframe
    df = pd.DataFrame()

    #set date column of dataframe
    df['sasdate'] = sasdate_index[:len(series)]

    #enter original series values in 'y' column
    df['y'] = series

    #set date column as index
    df = df.set_index('sasdate')

    #sort values to make sure they're in the right order
    df = df.sort_values('sasdate')

    #create columns of lagged values in for loop
    for i in range(1,lags+1):
        df['d'+str(i)] = df.y.shift(i)

    #return dataframe and exclude the first rows containing null values
    return df[lags:]

def hStepForward(data,model,h, sasdate_index):
    """
    :param train_frame: training dataframe
    :param model: fitted model
    :param h: steps to predict
    :param sasdate_index: index of original series
    :return: series of predicted values
    """
    #repeat for each forecasting step
    for j in range(h):

        #input data for next prediction
        x_in = data.tail(1).iloc[:,:-1].values

        #prediction
        y_hat = model.predict(x_in)

        #create dataframe of prediction with prediction and independent variables, one row
        df_tmp = pd.DataFrame(np.append(y_hat, x_in)).T

        #add column names
        df_tmp.columns = data.columns

        #set index
        df_tmp['sasdate'] = sasdate_index[-h+j]
        df_tmp = df_tmp.set_index('sasdate')

        #append to training data
        data = data.append(df_tmp)

    return data.y[-h:]


def FitMLP(sizes, act,max_iter,lags,train_series,sasdate_index, test_series, rndseed, alpha):
    """
    :param sizes: tuple of hidden layer sizes
    :param act: activation function, one of 'relu', 'logistic', 'tanh'
    :param max_iter: see documentation
    :param lags: lags for the model to use
    :param series: series to use
    :param n_ahead: number of steps to predict and evaluate
    :return: dictionary of para
    """
    #initialize scaler to scale values to unit interval
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    #transform input series
    train_series = scaler.fit_transform(train_series.values.reshape(-1, 1))

    #create training dataframe
    train_frame = CreateTrainFrame(train_series, lags, sasdate_index)

    #set random seed
    np.random.seed(rndseed)

    #initialize MLPRegressor
    #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    mlp = MLPRegressor(hidden_layer_sizes=(sizes), activation=act, max_iter=max_iter, alpha = alpha)

    #fit
    fit = mlp.fit(train_frame.iloc[:, 1:].values, train_frame['y'].values)

    #predict h steps ahead
    y_pred_mlp = hStepForward(train_frame, fit, len(test_series), sasdate_index)

    #rescale predictions to original scale
    y_pred_mlp = scaler.inverse_transform(y_pred_mlp.values.reshape(-1, 1))

    #create output dictionary
    dct_out= {'sizes':sizes, 'act':act, 'max_iter':max_iter,'alpha' : alpha, 'lags':lags,\
              'mse':mean_squared_error(test_series, y_pred_mlp), 'y_pred_mlp':y_pred_mlp}
    return dct_out

def EvaluateOOS(series, endpoint,n_ahead, max_nodesize,max_depth,max_lags, actfuncs, trainshare_mlp, rndseed, n_models = None, stepsize_nodes = 1,multiproc = True):
    """
    :param series: entire input series
    :param endpoint: endpoint until which series should be used
    :param n_ahead: number of steps to predict
    :param max_nodesize: maximum nodesize per layer
    :param max_depth: maximum number of layers for mlp
    :param max_lags: maximum number of lags
    :param actfuncs: list of activation functions
    :param trainshare_mlp: share of training data for mlp
    :param rndseed: random seed
    :param n_models: number of models to fit in case of random grid search
    :param multiproc: Boolen, if multiprocessing is allowed or not, defaults to True
    :return: dictionary of prediction results
    """
    #set random seed for random module
    random.seed(rndseed)

    #select series up until endpoint
    series = series[:endpoint]

    #split into train and validation set
    train = series[:-n_ahead]
    validation = series[-n_ahead:]

    #save index of series to variable for later indexing
    sasdate_index = series.index

    ###Optimize MLP Model
    # split series for in sample mlp validation, according to trainshare_mlp
    n_train_mlp = int(np.round(len(train) * trainshare_mlp))
    train_mlp = train[:n_train_mlp]
    test_mlp = train[n_train_mlp:]

    # initialize list of all possible combinations of nodesizes
    nodesizes = NodesizeCombinations(max_nodesize, max_depth, n_models, stepsize_nodes)

    #spread out grid for grid search
    ls_params = []
    for i in nodesizes:
        for func in actfuncs:
            #alpha (L2 regularization parameter) is initialized on log scale
            for alpha in list(np.round(np.logspace(np.log10(0.0001),np.log10(0.05),num = 5), 4)):
                ls_params.append([i, func, alpha])

    #sample n_models parameter combinations if n_models is given
    if n_models != None:
        ls_params = random.sample(ls_params, n_models)

    #Multiprocessing
    ls_out = []

    if multiproc == True:

        ls_out = Parallel(n_jobs = -1)(delayed(FitMLP)(*[param[0], param[1], 2000, max_lags, train_mlp, sasdate_index, test_mlp, rndseed, param[2]]) for param in ls_params)

    else:
        for param in ls_params:
            ls_out.append(FitMLP(param[0], param[1], 2000, max_lags, train_mlp, sasdate_index, test_mlp, rndseed, param[2]))

    #sort dataframe containing results and return the first results
    df_tmp = pd.DataFrame(ls_out).sort_values('mse').head(1)

    #fit best mlp model
    result = FitMLP(df_tmp.sizes.values[0], df_tmp.act.values[0], df_tmp.max_iter.values[0],
                                    df_tmp.lags.values[0], train, sasdate_index, validation, rndseed, df_tmp.alpha.values[0])

    #initialize dictionary for final output
    dct_fin = {}
    dct_fin['endpoint'] = endpoint
    dct_fin['MSE_MLP'] = mean_squared_error(result['y_pred_mlp'],validation.values)
    dct_fin['sizes'] = df_tmp.sizes.values[0]
    dct_fin['act'] = df_tmp.act.values[0]
    dct_fin['alpha'] = df_tmp.alpha.values[0]

    ###ARMA(0,0,0)
    #https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
    warnings.filterwarnings("ignore")
    model = ARIMA(train, order=(0,0,0))
    model_fit = model.fit()
    forecast = model_fit.predict(start=validation.index[0], end=validation.index[-1])
    warnings.filterwarnings('default')
    dct_fin['MSE_ARMA(0,0)'] = mean_squared_error(forecast.values,validation.values)

    ###ARMA(p,q)
    #select optimal model parameters using autoarima
    autoarima = auto_arima(train, start_p=0, d=0, start_q=0, max_p=max_lags, max_d=1, max_q=max_lags)

    #retrain model using statsmodels ARIMA
    warnings.filterwarnings("ignore")
    model = ARIMA(train, order=(autoarima.order[0], 0, autoarima.order[2]))
    model_fit = model.fit()
    forecast = model_fit.predict(start=validation.index[0], end=validation.index[-1])
    warnings.filterwarnings('default')
    dct_fin['MSE_ARMA(p,q)'] = mean_squared_error(forecast.values, validation.values)
    dct_fin['p'] = autoarima.order[0]
    dct_fin['q'] = autoarima.order[2]
    dct_fin['n_ahead'] = n_ahead
    dct_fin['max_lags'] = max_lags
    dct_fin['trainshare_mlp'] = trainshare_mlp
    dct_fin['rndseed'] = rndseed
    dct_fin['n_models'] = n_models

    return dct_fin


if __name__ == '__main__':

    ################################################################################
    #CONTROL PANEL
    ################################################################################

    #maximum nodesize for each layer
    max_nodesize = 100

    #maximum number of layers
    max_depth = 5

    #earliest endpoint up to which to run the benchmark
    earliest_endpoint = 692

    # list of activation functions to use in Grid Search
    actfuncs = ['relu']

    #share of training data vs test data for mlp optimization
    trainshare_mlp = 0.9

    #maximum number of lags
    max_lags = 100

    #global random seed
    rndseed = 42

    #number of models to estimate per iteration
    n_models = 2000

    #stepsize for model selection
    stepsize_nodes = 5
    ################################################################################

    #read in data
    df = pd.read_csv('../data/current.csv')

    #drop rows that contain exclusively null values
    df = df.dropna(how='all')

    #drop first row, as first row contains info on transformations, not real data
    df = df.drop(0)

    #define date column
    df.sasdate = pd.to_datetime(df['sasdate'])

    #set date column as index
    df = df.set_index('sasdate')

    #save index in variable for later indexing purposes
    sasdate_index = df.index

    start = time.perf_counter()

    series_name = 'HOUSTMW'
    n_ahead = 5

    for n_ahead in [1,5, 10]:
            ls_out = []
            # define series to use
            series = df[series_name]
            print(f'{series_name}, steps predicted: {n_ahead}')

            #loop backwards through the dataframe and cut 1 more observation each time, until we reach earliest endpoint
            for i in tqdm(range(len(df), earliest_endpoint, -1)):
                rndseed+=1
                #Run OOS prediction Evaluation for each step and append results
                ls_out.append(
                    EvaluateOOS(series, df.index[i-1], n_ahead, max_nodesize, max_depth, max_lags, actfuncs, trainshare_mlp, rndseed, n_models, stepsize_nodes))
                finish = time.perf_counter()

            #put results into dataframe
            df_res = pd.DataFrame(ls_out)

            print(df_res.head())
            print(df_res[[i for i in df_res.columns if 'MSE' in i]].mean())
            #generate unique identifier to associate benchmark metadata
            identifier = uuid.uuid4().hex

            #save output dataframe
            str_filename = '../results/' + f'results_series_name_{series_name}_n_ahead_{n_ahead}_earliest_endpoint_{earliest_endpoint}_{identifier}.pkl'

            filehandler = open(str_filename, 'wb')
            pickle.dump(df_res, filehandler)

            filehandler.close()

            #initialize dictionary for benchmark metadata
            dct_res = {}
            dct_res['max_nodesize'] = max_nodesize
            dct_res['max_depth'] = max_depth
            dct_res['actfuncs'] = actfuncs
            dct_res['trainshare_mlp'] = trainshare_mlp
            dct_res['max_lags'] = max_lags
            dct_res['rndseed'] = rndseed
            dct_res['n_models'] = n_models
            dct_res['stepsize_nodes'] = stepsize_nodes

            str_filename_dct = '../results/' + f'{identifier}.pkl'
            filehandler = open(str_filename_dct, 'wb')
            pickle.dump(dct_res, filehandler)

            filehandler.close()
            print(str_filename)

    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds')