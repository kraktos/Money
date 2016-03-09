import math
import sys, os
import logging
import pandas as pd
import numpy as np

import core_ml.TimeSeriesNnet as ts

# define loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# the core NNet block
def get_time_series_predictions(time_series):
    logger.debug('\tTime series is of length %s', len(time_series))
    logger.debug('%s', time_series)

    if len(time_series) < 5:
        return None, None
    else:
        min_val = time_series.min()
        if min_val < 0:
            time_series = time_series.apply((lambda x: x + abs(min_val)))

    # iteratively try to find the best configuration
    # hold out some n-points and predict those, compute prediction error and
    # change model params to minimize error
    error, lag, neural_net = perform_parameter_optimization(time_series)

    time_series_standard_deviation = np.std(time_series)

    last = time_series[-1:]
    last_value = last.values[0]

    conf_30_max = last_value + time_series_standard_deviation * 0.38
    conf_30_min = last_value - time_series_standard_deviation * 0.38

    conf_50_max = last_value + time_series_standard_deviation * 0.67
    conf_50_min = last_value - time_series_standard_deviation * 0.67

    conf_80_max = last_value + time_series_standard_deviation * 1.28
    conf_80_min = last_value - time_series_standard_deviation * 1.28

    conf_90_max = last_value + time_series_standard_deviation * 1.64
    conf_90_min = last_value - time_series_standard_deviation * 1.64

    conf_95_max = last_value + time_series_standard_deviation * 1.96
    conf_95_min = last_value - time_series_standard_deviation * 1.96

    if neural_net:
        # fit the data into the best  model
        # while final prediction is way off the charts, try twice more, else do not generate card
        for i in range(0, 2):
            neural_net.fit(time_series, lag=lag+1, epochs=6000 + 2*i)
            # use the model to predict
            final_pred = neural_net.predict_ahead(n_ahead=1)

            if final_pred[0] > conf_95_max:
                if not math.isnan(final_pred):
                    with open('{}/user_error_conf_logs.tsv'.format(os.getcwd()), 'a') as the_file:
                        the_file.write(
                            '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(final_pred[0], conf_30_min, conf_30_max,
                                                                                  conf_50_min, conf_50_max,
                                                                                  conf_80_min, conf_80_max,
                                                                                  conf_90_min, conf_90_max,
                                                                                  conf_95_min, conf_95_max))
            else:
                if min_val < 0:
                    final_pred[0] = final_pred[0] - abs(min_val)
                return error, final_pred

    return None, None


# take the time series data set, and optimize the params to get the minimum error
def perform_parameter_optimization(time_series):
    error = sys.float_info.max
    local_run_error = sys.float_info.max

    best_model = None
    activation_functions = ['relu', 'hard_sigmoid', 'tanh', 'sigmoid', 'softmax', 'softplus']

    k = len(time_series) - 1

    # for k in range(4, len(time_series)):
    train_ts_data = time_series[0:k:1]
    test_ts_data = time_series[k: k + 1: 1]

    # define lag
    lag = len(train_ts_data) - 10

    # define nodes, rule of thumb, the number of hidden layer nodes is roughly the
    # mean of the input and output layer node sizes
    # in this case, the lag determines the number of nodes in the input layer
    # and the output is the number of days ahead (also 1 in this case) and allow at least 2 nodes in hidden layer
    hidden_layer_node_count = int(max(2, int(math.ceil((lag + 1) / 2))))

    nodes_in_hidden_layer = max(3, hidden_layer_node_count)

    # if there is no model then try finding it, else reuse it
    # if not best_model:

    # perform a grid search
    logger.info('Grid Searching parameters..')

    # for nodes_in_hidden_layer in range(2, max(3, hidden_layer_node_count), 1):
    for indices in activation_functions:
        func_l1 = indices
        # func_l2 = indices[1]
        # func_l3 = indices[2]

        # generate the neural net
        neural_net = ts.TimeSeriesNnet(
            hidden_layers=[nodes_in_hidden_layer],
            activation_functions=[func_l1])

        # fit the data into the model
        neural_net.fit(train_ts_data, lag=lag, epochs=10)

        # get the predictions
        predictions = neural_net.predict_ahead(n_ahead=len(test_ts_data))
        if not np.isinf(predictions):
            actual = np.array(test_ts_data, dtype=pd.Series)[0]
            actual = actual if actual > 0 else 1
            pred = predictions[0]

            # computing MAPE
            local_run_error = (abs(actual - pred) * 100 / actual)

        if local_run_error == 0:
            best_model = neural_net
            error = local_run_error
            break
        elif error > local_run_error:
            error = local_run_error
            best_model = neural_net

    # open this to get more accurate errors, essentially increases the epoch,

    # get the best error with the best model
    # best_model.fit(train_ts_data, lag=lag, epochs=100)
    # predictions = best_model.predict_ahead(n_ahead=len(test_ts_data))
    # logger.info('Finding least error..')

    # if not np.isinf(predictions):
    #     actual = np.array(test_ts_data, dtype=pd.Series)[0]
    #     actual = actual if actual > 0 else 1
    #     pred = predictions[0]
    #     error = (abs(actual - pred) * 100 / actual)
    #     if error <= 100:
    #         logger.info("MAPE = %s", error)
    #         with open('{}/user_best_error_logs.tsv'.format(os.getcwd()), 'a') as the_file:
    #             the_file.write('{}\t{}\t{}\t{}\n'.format(error, len(train_ts_data), len(test_ts_data),
    #                                                      len(time_series)))

    return error, lag, best_model
