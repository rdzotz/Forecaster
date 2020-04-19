# %%
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
from scipy import fftpack
from sklearn.metrics import r2_score

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

class forecasterNN:
  '''Forecaster includes helper functions

  Notes
  -----
  Write up class into object oriented structure. 
  '''

  # def __init__(self):
  
  def dataGet(folderPath, split=None, unstack=True):
      ''' Reads ``*.csv`` data from ``folderPath``, concatenates into single
      Pandas.series.

      Parameters
      ----------
      folderPath : str
          The path to the folder containing in.
      split : int (default is None)
          index to split into training and test data

      Returns
      -------
      train : np.array
          training data
      test : np.array/ None
          test data
      '''
      
      files = glob.glob(os.path.join(folderPath, "*.csv"))
      files.sort()
      
      dfs = []
      
      for file in files:
          df = pd.read_csv(file, index_col=None, header=None)
          dfs.append(df)
          
      if unstack:
        data = np.reshape(pd.concat(dfs, axis=0).stack().reset_index(drop=True).values, 
                          (-1,1))
      else:
        data = pd.concat(dfs, axis=0).reset_index(drop=True).values
      
      if split:
          test = data[split:]
          train = data[:split]
      else:
          train = data
          test = None
         
      return train, test

  def dataFFT(x, title=None, f_s=1):
    '''
    Parameters
    ----------
    x : np.array
      dataset to plot
    title : str (default = None)
      The title of the data to be plotted.
    f_s : int (default =1)
      Labelled value/values.

    Returns
    -------
    None
    '''
    t = np.arange(0,len(x))

    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal amplitude')
    X = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x)) * f_s

    fig, ax = plt.subplots()

    ax.plot(freqs, np.abs(X))
    ax.set_title(title)
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')

  def best_fit(xs,ys):
    '''Calculate the line of best fit.
    '''
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
          ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
      
    b = np.mean(ys) - m*np.mean(xs)
      
    return m, b

  def dataStats(data, plot=True, title=None, verbose=False, standadize=False):
    '''Calculate some basic statistics on the input data. 
    The columnwise mean, std., and line of best fit.

    Parameters
    ----------
    data : np.array 2D
      Columns are assumed to represent each datapoint in months.
    plot : bool
      Plot the 
    title : str (Default None)
      The title of the plot
    standadize : bool (Default False)
      Standadize the distribuition of mean and std. 

    Returns
    -------
      bestLine :  dict
        The slope m and intercept b of the line of best fit 
        for both the mean and std.
    '''
    meanM = data.mean(axis=0)
    stdS = data.std(axis=0)

    if standadize:
      meanM = (meanM - meanM.mean(axis=0))/meanM.std(axis=0)
      stdS = (stdS - stdS.mean(axis=0))/stdS.std(axis=0)

    x = np.arange(1, data.shape[1]+1)

    # Calculate the line of best fit for mean
    m_M, b_M = forecasterNN.best_fit(x, meanM)
    fitLineM = [(m_M*xi) + b_M for xi in  x]
    # mse_M = np.mean((meanM - fitLineM)**2) 
    R2_M = r2_score(meanM, fitLineM)

    # Calculate the line of best fit for var
    m_S, b_S = forecasterNN.best_fit(x, stdS)
    fitLineS = [(m_S*xi) + b_S for xi in  x]
    R2_S = r2_score(stdS, fitLineS)

    statsDict = dict(x = x,
                     meanM = meanM,
                     stdS = stdS,
                     m_M = m_M, b_M = b_M,
                     m_S = m_S, b_S = b_S,
                     R2_M = R2_M,
                     R2_S = R2_S)

    for key in statsDict:
      if key not in ['meanM','stdM'] and verbose:
        print('The %s =' %key, statsDict[key], '\n')

    if plot:
      fig, ax = plt.subplots()
      ax.plot(x, meanM, 'o')
      ax.plot(x, fitLineM)
      ax.set_title(title)
      ax.set_xlabel('Month')
      ax.set_ylabel('Mean')
      ax.text(x[3], fitLineM[3], ' R2 = %g' %R2_M)

      fig, ax = plt.subplots()
      ax.plot(x, stdS, 'o')
      ax.plot(x, fitLineS)
      ax.set_title(title)
      ax.set_xlabel('Month')
      ax.set_ylabel('std.')
      ax.text(x[3], fitLineS[3], ' R2 = %g' %R2_S)

    return statsDict

  def baselinePlot(data, forecast, title=None, standadize=False, 
                  NNmean=None, NNstd=None):
      '''Plot the baseline prediction based on a simple extrapolation of the 
      mean and varience extracted from the columns of ``data``. Note this function 
      first standadizes the data in order to make a comparison with the standadised 
      output of typical neural networks

      Parameters
      ----------
      data : ndarray
        The raw data including the focast.
      forecast : int
        The number of steps to forecast.
      standadize : bool (Default False)
        Standadize the distribuition of mean and std. 
      NNmean : float, (Default None)
        Alternative forecaster mean
      NNstd : float, (Default None)
        Alternative forecaster std.

      Returns
      -------
      error : float
        the squared error percentage.
      '''

      data

      dataStatBase = forecasterNN.dataStats(data, plot=False, standadize=False)

      dataStatHist = forecasterNN.dataStats(data[:,:-forecast], plot=False, standadize=False)

      months = data.shape[1] -forecast

      x_pred = dataStatBase['x']

      fitLinePred_mean = [(dataStatHist['m_M']*xi) + dataStatHist['b_M'] for xi in  x_pred]
      fitLinePred_var = [(dataStatHist['m_S']*xi) + dataStatHist['b_S'] for xi in  x_pred]

      for line, var in zip([fitLinePred_mean, fitLinePred_var],['meanM', 'stdS']):
        fig, ax = plt.subplots()
        ax.plot(dataStatHist['x'], dataStatHist[var], 'o', label='Raw Data')
        ax.plot(x_pred, line, label='Fitted Line %g Months' %months)
        ax.plot(x_pred[-1], line[-1], 'ro', label='Forecast to %g th Month' %data.shape[1])
        if NNmean and var=='meanM':
          ax.plot(x_pred[-1], NNmean, 'r*', label='RNN forecast')
        if NNstd and var=='stdS':
          ax.plot(x_pred[-1], NNstd, 'r*', label='RNN forecast')
        ax.plot(x_pred[-1], dataStatBase[var][-1], 'go', label='True Value')
        ax.set_title(title)
        ax.set_xlabel('Month')
        ax.set_ylabel(var)
        ax.legend()
      
      error_mean = (dataStatBase['meanM'][-1] - fitLinePred_mean[-1])**2 
      error_std = (dataStatBase['stdS'][-1] - fitLinePred_var[-1])**2 

      if NNmean:
        error_meanNN = (dataStatBase['meanM'][-1] - NNmean)**2 
        print('The squared error for %s mean RNN is %0.3g' %(title, error_meanNN))
      if NNstd:
        error_stdNN = (dataStatBase['stdS'][-1] - NNstd)**2 
        print('The squared error for %s std. RNN is %0.3g' %(title, error_stdNN))
      # error_mean = mean_squared_error(dataStatBase['meanM'][-1], fitLinePred_mean[-1])
      # error_std = mean_squared_error(dataStatBase['stdS'][-1], fitLinePred_var[-1])
      print('The squared error for %s mean is %0.3g' %(title, error_mean))
      print('The squared error for %s std. is %0.3g' %(title, error_std))
      return error_mean, error_std

  def dataPreprocess(dataset, target, start_index, end_index, window_width,
                        target_size, step, stride, single_step=False):
      ''' This function structures the input data into windows for input into 
      the RNN, though could be replaced with the tf.data.Dataset.window function.
      
      Parameters
      ----------
      dataset : np.array
          Entire dataset
      target : float/array
          Labelled value/values.
      start_index : int
          The index from which values in ``dataset`` will be included.
      end_index : int
          The index to which values in ``dataset`` will be included.
      window_width : int
          The width of the window which is used to generate each forecast.
      target_size : int
          The advancement of the forecast.
      step : int
          The sampling density within each window used to generate the data.
      stride : int
          The advancement of the windows in no. of indicies
      single_step : bool, optional (default is False)
          If True only a single label will be output.

      Returns
      -------
      data, labels

      '''
      data = []
      labels = []
      
      if step >= window_width:
          raise Warning('The step size %g >= the window width %g' 
                        %(step, window_width))

      start_index = start_index + window_width
      if end_index is None:
          end_index = len(dataset) - target_size
      
      for i in range(start_index, end_index, stride):
          indices = range(i-window_width, i, step)
          data.append(dataset[indices])
      
          if single_step:
            labels.append(target[i+target_size])
          else:
            #labels.append(target[slice(i,target_size,step)])
            labels.append(target[i:i+target_size])
      
      return np.array(data), np.array(labels)


  def create_time_steps(length):
    return list(range(-length, 0))

  def dataTrain(data, rolling_window, down_sample, target_size, 
                window_width_fractor, split, stat, STEP=1, stride=1, 
                BATCH_SIZE = 64, BUFFER_SIZE = 1000, verbose=True):
    '''Set Up the model parameters. Note all index related imputs should be in 
    the dimensions of the original ``data`` array.

    Parameters
    ----------
    data : np.array()
      The array containing a single series.
    rolling_window : int
      The number of elements from which the rolling window will be calculated.
    down_sample : int
      Every other sample, downsampling inteded to reduce training time.
    target_size : int
      The size in terms of sample points which the forecaster much generate.
    window_width_fractor : int
      The width of the previous measurements in terms of 
    split : int
      The index at which the training data should be separated from the test data.
      Note the test data will be cast into the validation set to be used for
      cross-validation after each BATCH.
    stat : str
      expects "mean" or "std".
    STEP : int (default=1)
      The downsampling used in each window.
    stride : int (default=1)
      The number of indicies by which each window advances to generate each traing
      example.
    BATCH_SIZE : int (default=64)
      The number of training examples used in each epoch
    BUFFER_SIZE : int (default=100)
      The buffer from which random examples are shuffled. Should ideally be equal
      to the number of examples in ``data``.
    verbose : bool (default=True)
      Controls the verbosity of the function.
    trainScale, valScale : sklearn scaler
      The training and validation set scaler. Intended to be reapplied after 
      training or evaluation.

    returns
    -------
    train_data, val_data:
      Training and validation data in tf.dataset
    '''

    # Create pandas dataframe
    data = pd.DataFrame(data)

    # Rolling window over the dataset
    if stat =='mean':
      data_roll = data.rolling(window=rolling_window, center=False).mean().dropna()
    elif stat =='std':
      data_roll = data.rolling(window=rolling_window, center=False).std().dropna()

    # Downsample the dataset
    data_roll_samp = data_roll[::down_sample]

    # The target size of the prediction in downsampled units
    target_size_samp = np.arange(0, target_size, down_sample).shape[0]

    # Define window size input to be used in training for each prediction
    window_width = target_size_samp*window_width_fractor

    # Remove the final month
    data_train = data_roll_samp.loc[data_roll_samp.index < split]
    data_val = data_roll_samp.loc[data_roll_samp.index >= split- target_size*window_width_fractor-1]


    # Standadize the data using only the training data
    # data = (data - data.loc[:split].mean(axis=0))/data.loc[:split].std(axis=0)
    trainScale = StandardScaler()
    valScale = StandardScaler()
    trainScale = trainScale.fit(data_train)
    valScale = valScale.fit(data_val)
    data_train = trainScale.transform(data_train)
    data_val = valScale.transform(data_val)
    # data_train = (data_train - data_train.mean(axis=0))/data_train.std(axis=0)
    # data_val = (data_val - data_val.mean(axis=0))/data_val.std(axis=0)

    
    if verbose:
      print('data_train: ',data_train.shape)
      print('data_val: ',data_val.shape)
      print('window width: ',window_width)
      print('Target size: ',target_size_samp)

    # Create the training and validation set
    x_train, y_train = forecasterNN.dataPreprocess(data_train, data_train[:, 0], 0,
                                                    None, window_width,
                                                    target_size_samp, STEP, stride)

    # Create the training and validation set
    x_val, y_val = forecasterNN.dataPreprocess(data_val, data_val[:, 0], 0,
                                                    None, window_width,
                                                    target_size_samp, STEP, stride)
    
    if verbose:
      print ('The training data dims : {}'.format(x_train.shape))
      print ('The training label data dims : {} \n'.format(y_train.shape))

      print ('The validation data dims : {}'.format(x_val.shape))
      print ('The validation label data dims : {}\n'.format(y_val.shape))

      print ('Single window of past data dims : {}'.format(x_train[0].shape))
      print ('Target data to predict dims: {}\n'.format(y_train[0].shape))

    # %% Cast into tf.data.Dataset for training
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE, seed=51).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    if verbose:
      print('A single training example')
      # %% Have a look at a single training example
      for x, y in train_data.take(1):
        forecasterNN.plot_forecast(x[0], y[0], np.array([0]), step=STEP)

    return train_data, val_data, BATCH_SIZE, target_size_samp, trainScale, valScale

  def dataReform(modelM, modelS, val_dataM, val_dataS, valScaleM, valScaleS, down_sample, numSamp):
    '''Reconstruction of the output data into the original data. This involves
    both interpelation, and inverse transformation in terms of scale.

    Parameters
    ----------
    modelM : tf.model
      The pre-trained model for the from which the forecast of the mean can 
      be made.
    modelS : tf.model
      The pre-trained model for the from which the forecast of the std. can 
      be made.
    val_dataM : tf.databse
      The validation data for the mean.
    val_dataS : tf.databse
      The validation data for the std.
    valScaleM/valScaleM : sklearn scale object
      The scaler for the mean/std. validation data.
    down_sample : int
      The downsampling factor by which the data will be interperlated.
    numSamp : int
      The number of desiered samples within the reconstructed series.
    '''
    for x, y in val_dataM1.take(1):
      meanforecast = modelM1.predict(x)[0]

    for x, y in val_dataS1.take(1):
      stdforecast = modelS1.predict(x)[0]
    
    meanInt = np.interp(np.arange(0,numSamp,1), np.arange(0,numSamp,down_sample), meanforecast)
    stdInt = np.interp(np.arange(0,numSamp,1), np.arange(0,numSamp,down_sample), stdforecast)

    meanInt = valScaleM.inverse_transform(meanInt)
    stdInt = valScaleS.inverse_transform(stdInt)

    # Reconstruction
    forecast = np.random.normal(meanInt, stdInt)

    return forecast, meanInt.mean(), stdInt.mean()

  def plot_forecast(history, true_future, prediction, step):
    plt.figure(figsize=(12, 6))
    num_in = forecasterNN.create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history), label='History')
    plt.plot(np.arange(num_out)/step, np.array(true_future), 'bo',
            label='True Future')
    if prediction.any():
      plt.plot(np.arange(num_out)/step, np.array(prediction), 'ro',
              label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

  def plot_train_history(history, title=None):
    '''Plot the training histroy 
    Parameters
    ----------
    history : tensorflow.python.keras.callbacks
        Output class from model.fit.
    title : str (default None)
      The title of the plot
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
