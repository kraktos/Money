import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import StandardScaler
from keras.callbacks import BaseLogger


class TimeSeriesNnet(object):
    def __init__(self, hidden_layers=[20, 15, 5], activation_functions=['sigmoid', 'sigmoid', 'sigmoid']):
        self.hidden_layers = hidden_layers
        self.activation_functions = activation_functions

        if len(self.hidden_layers) != len(self.activation_functions):
            raise Exception("hidden_layers size must match activation_functions size")

    def fit(self, timeseries, lag=7, epochs=10000, verbose=0, optimizer='sgd'):
        self.timeseries = np.array(timeseries, dtype="float64")  # Apply log transformation por variance stationarity
        self.lag = lag
        self.y = None
        self.n = len(timeseries)
        if self.lag >= self.n:
            raise ValueError("Lag is higher than length of the timeseries")
        self.X = np.zeros((self.n - self.lag, self.lag), dtype="float64")
        with np.errstate(divide='ignore'):
            self.y = np.log(self.timeseries[self.lag:])

        self.epochs = epochs
        self.scaler = StandardScaler()
        self.verbose = verbose
        self.optimizer = optimizer

        # Building X matrix
        for i in range(0, self.n - lag):
            self.X[i, :] = self.timeseries[range(i, i + lag)]
        #
        # if verbose:
        #     print "Scaling data"
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)

        # Neural net architecture
        self.nn = Sequential()
        self.nn.add(Dense(self.X.shape[1], self.hidden_layers[0]))
        self.nn.add(Activation(self.activation_functions[0]))
        self.nn.add(Dropout(0.25))

        for i, layer in enumerate(self.hidden_layers[:-1]):
            self.nn.add(Dense(self.hidden_layers[i], self.hidden_layers[i + 1]))
            self.nn.add(Activation(self.activation_functions[i]))
            self.nn.add(Dropout(0.25))

        # Add final node
        self.nn.add(Dense(self.hidden_layers[-1], 1))
        self.nn.compile(loss='mean_absolute_error', optimizer=self.optimizer)

        # Train neural net
        self.nn.fit(self.X, self.y, nb_epoch=self.epochs, verbose=self.verbose,
                    callbacks=[BaseLogger()])

    def predict(self):
        # Doing weird stuff to scale *only* the first value
        self.next_X = np.concatenate((np.array([self.y[-1]]), self.X[-1, :-1]), axis=0)
        self.next_X = self.next_X.reshape((1, self.lag))
        self.next_X = self.scaler.transform(self.next_X)
        self.valid_x = self.next_X[0, 0]
        # Doing it right now
        self.next_X = np.concatenate((np.array([self.valid_x]), self.X[-1, :-1]), axis=0)
        self.next_X = self.next_X.reshape((1, self.lag))
        self.next_y = self.nn.predict(self.next_X)
        return np.exp(self.next_y)

    def predict_ahead(self, n_ahead=1):
        # Store predictions and predict iteratively
        self.predictions = np.zeros(n_ahead)

        for i in range(n_ahead):
            self.current_x = self.timeseries[-self.lag:]
            self.current_x = self.current_x.reshape((1, self.lag))
            self.current_x = self.scaler.transform(self.current_x)
            self.next_pred = self.nn.predict(self.current_x)
            self.predictions[i] = np.exp(self.next_pred[0, 0])
            self.timeseries = np.concatenate((self.timeseries, np.exp(self.next_pred[0, :])), axis=0)

        return self.predictions
