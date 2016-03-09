__author__ = 'arnabdutta'

import Quandl
import pandas as pd
import matplotlib.pyplot as mpl

from core_ml.ml_stack import get_time_series_predictions

time_series_df = Quandl.get("WIKI/AAPL")
print(type(time_series_df))

column_name = 'Open'
error, predicted_balance = get_time_series_predictions(pd.Series(time_series_df[column_name].tolist(),
                                                                             name=column_name))

print (predicted_balance)
print (error)

y_val = time_series_df[column_name]
y_val[len(time_series_df)-1] = predicted_balance[0]

fig = mpl.figure()
y_val.plot()

mpl.gcf().autofmt_xdate()
mpl.savefig("MARKET_.png")
mpl.close(fig)

