import anfis
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from yahoofinancials import YahooFinancials
from pandas.plotting import register_matplotlib_converters
from datetime import datetime
import math
import os
from membership import membershipfunction
here = os.path.dirname(os.path.abspath(__file__))


yahoo_financials = YahooFinancials('LPL')

data = yahoo_financials.get_historical_price_data(start_date='2020-01-01',
                                                  end_date=datetime.today().strftime('%Y-%m-%d'),
                                                  time_interval='daily')

tsla_df = pd.DataFrame(data['LPL']['prices'])
tsla_df = tsla_df.drop('date', axis=1).set_index('formatted_date')
print(tsla_df.head())
y = tsla_df['close'].astype(float)
x = tsla_df.drop(['close'],axis=1)


print (x.head())
def sigmaCalc(a,b,c,d,e):
    Len = len(a)
    result = [[0]*2 for _ in range(5)]
    print(sum(a),sum(b))
    result[0][0] += sum(a)/Len
    result[1][0] += sum(b)/Len
    result[2][0] += sum(c)/Len
    result[3][0] += sum(d)/Len
    result[4][0] += sum(e)/Len
    print(result)
    for i in range(Len):
        result[0][1] += (a[i] - result[0][0])**2
        result[1][1] += (b[i] - result[1][0])**2
        result[2][1] += (c[i] - result[2][0])**2
        result[3][1] += (d[i] - result[3][0])**2
        result[4][1] += (e[i] - result[4][0])**2
    result[0][1] = math.sqrt(result[0][1] / (Len - 1))
    result[1][1] = math.sqrt(result[1][1] / (Len - 1))
    result[2][1] = math.sqrt(result[2][1] / (Len - 1))
    result[3][1] = math.sqrt(result[3][1] / (Len - 1))
    result[4][1] = math.sqrt(result[4][1] / (Len - 1))
    print(result)
    return result

a = x['high']
b = x['low']
c = x['open']
d = x['volume']
e = x['adjclose']

print (len(a))


result3 = sigmaCalc(a[:128],b[:128],c[:128],d[:128],e[:128])
result1 = sigmaCalc(a[128:256],b[128:256],c[128:256],d[128:256],e[128:256])
result2 = sigmaCalc(a[256:383],b[256:383],c[256:383],d[256:383],e[256:383])
result4 = sigmaCalc(a[:383],b[:383],c[:383],d[:383],e[:383])
"""""
[[['gaussmf',{'mean':52.225,'sigma':12.494}],['gaussmf',{'mean':60.061,'sigma':8.857}],['gaussmf',{'mean':374.025,'sigma':266.568}]],
      [['gaussmf',{'mean':50.651,'sigma':12.273}],['gaussmf',{'mean': 57.822,'sigma':8.53}],['gaussmf',{'mean':355.801,'sigma':255.204}]],
      [['gaussmf',{'mean':51.481,'sigma':12.423}],['gaussmf',{'mean':58.964,'sigma':8.671}],['gaussmf',{'mean':365.339,'sigma':261.446}]],
      [['gaussmf',{'mean':27021089.632,'sigma':14138571.11}],['gaussmf',{'mean':43347931.965,'sigma':23808837.256}],['gaussmf',{'mean':57679820.086,'sigma':37664847.262}]],
      [['gaussmf',{'mean':51.449,'sigma':12.396}],['gaussmf',{'mean':58.988,'sigma':8.703}],['gaussmf',{'mean':365.544,'sigma':261.316}]]]
      
      [[['gaussmf',{'mean':56.229,'sigma':11.994}],['gaussmf',{'mean':268.131,'sigma':264.411}]],
      [['gaussmf',{'mean':54.385,'sigma':11.653}],['gaussmf',{'mean':255.276,'sigma':252.454}]],
      [['gaussmf',{'mean':55.335,'sigma':11.842}],['gaussmf',{'mean':262.003,'sigma':258.91}]],
      [['gaussmf',{'mean':31651303.597,'sigma':19270784.466}],['gaussmf',{'mean':53730486.599,'sigma':33988273.312}]],
      [['gaussmf',{'mean':55.326,'sigma':11.829}],['gaussmf',{'mean':262.143,'sigma':258.875}]]]
"""
# find how to calculate the gaussmf mean and sigma for the 5 variables, in order to have the membership functions to predict
mf = [[['gaussmf',{'mean':result4[0][0],'sigma':result4[0][1]}],['gaussmf',{'mean':result3[0][0],'sigma':result3[0][1]}]],
      [['gaussmf',{'mean':result4[1][0],'sigma':result4[1][1]}],['gaussmf',{'mean':result3[1][0],'sigma':result3[1][1]}]],
      [['gaussmf',{'mean':result4[2][0],'sigma':result4[2][1]}],['gaussmf',{'mean':result3[2][0],'sigma':result3[2][1]}]],
      [['gaussmf',{'mean':result4[3][0],'sigma':result4[3][1]}],['gaussmf',{'mean':result3[3][0],'sigma':result3[3][1]}]],
      [['gaussmf',{'mean':result4[4][0],'sigma':result4[4][1]}],['gaussmf',{'mean':result3[4][0],'sigma':result3[4][1]}]]]



mfc = membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(x, y, mfc)
plt.plot(anf.Y)
plt.show()
anf.trainHybridJangOffLine(epochs=20)

with open(os.path.join(here, "state.pickle"), "wb") as f:
    pickle.dump(anf,f)
    print("pickle has been written")

print("plotting errors")
anf.showErr()
print("plotting results")
anf.showRes()

