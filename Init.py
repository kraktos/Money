__author__ = 'arnabdutta'

import Quandl
import pandas as pd
import matplotlib.pyplot as mpl

from core_ml.ml_stack import get_time_series_predictions

SHARE_ID = 'OECD/HEALTH_STAT_CICDHOCD_TXCMILTX_GBR'
time_series_df = Quandl.get(SHARE_ID)
print(type(time_series_df))
print ()
column_name = 'Value'
error, predicted_balance = get_time_series_predictions(pd.Series(time_series_df[column_name].tolist(),
                                                                             name=column_name))

print (predicted_balance)
print (error)

y_val = time_series_df[column_name]
y_val[len(time_series_df)-1] = predicted_balance[0]

fig = mpl.figure()
y_val.plot()

mpl.gcf().autofmt_xdate()
mpl.title(SHARE_ID.replace(" ", "_").replace("/", "_"))
mpl.savefig("{}.png".format(SHARE_ID.replace(" ", "_").replace("/", "_")))
mpl.close(fig)

