import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()   
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
pd.set_option('precision',0)
import warnings
warnings.filterwarnings('ignore')
table = pd.read_csv('covid_19_data.csv',parse_dates=['ObservationDate'])
pd.set_option('display.max_rows', table.shape[0]) 
pd.set_option('display.max_columns', None)
table.style.set_properties(subset=['ad_description'], **{'width-max': '100px'})
table.head(10).style.background_gradient(cmap='cool')

table['Country/Region'].value_counts().head(20)
table.isnull().sum()

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']
table['Active'] = table['Confirmed'] - table['Deaths'] - table['Recovered']
table[['Province/State']] = table[['Province/State']].fillna('')
table[cases] = table[cases].fillna(0)
latest = table[table['ObservationDate'] == max(table['ObservationDate'])].reset_index()

latest_grouped = latest['Confirmed'] - latest['Deaths'] - latest['Recovered']
latest_grouped = latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

pred = latest_grouped.sort_values(by='Confirmed', ascending=False)
pred = pred.reset_index(drop=True)
cm = sns.light_palette("red", as_cmap=True)
pred.head(11).style.background_gradient(cmap=cm).background_gradient(cmap='Greens',subset=["Recovered"])\
.background_gradient(cmap='Blues',subset=["Active"]).background_gradient(cmap='Oranges',subset=["Confirmed"])



temp = table.groupby('ObservationDate')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = temp.sort_values('ObservationDate', ascending=False)
cm = sns.light_palette("red", as_cmap=True)
temp.head(11).style.background_gradient(cmap=cm).background_gradient(cmap='Greens',subset=["Recovered"]).background_gradient(cmap='Blues',subset=["Active"]).background_gradient(cmap='Oranges',subset=["Confirmed"])

total_count = pd.DataFrame({'Category':'Deaths', 'Count':temp.head(1)['Deaths']})
total_count = total_count.append({'Category':'Recovered','Count':int(temp.head(1)['Recovered'])}, ignore_index=True)
total_count = total_count.append({'Category':"Confirmed",'Count':int(temp.head(1)['Confirmed'])}, ignore_index=True)
total_count = total_count.append({'Category':"Active",'Count':int(temp.head(1)['Active'])}, ignore_index=True)
fig = px.bar(total_count, x='Count', y='Category',
             hover_data=['Count'], color='Count',
             labels={}, orientation='h',height=400, width = 650)
fig.update_layout(title_text='Confirmed vs Recovered vs Death cases vs Active')
fig.show()


ind_confirmed = table[table['Country/Region'] == 'India'].groupby(['ObservationDate'])['Confirmed'].sum().tolist()
ind_recovered = table[table['Country/Region'] == 'India'].groupby(['ObservationDate'])['Recovered'].sum().tolist()
ind_deaths = table[table['Country/Region'] == 'India'].groupby(['ObservationDate'])['Deaths'].sum().tolist()
plt.figure(figsize = (15,8))
plt.plot(ind_confirmed, color = 'c', marker = '.', label = 'Number of Cases')
plt.plot(ind_recovered, color = 'g', marker = '.', label = 'Recovered')
plt.plot(ind_deaths, color = 'r', marker = '.', label = 'Deaths')
plt.title('Covid-19 Cases in India')
plt.xlabel('Days')
plt.ylabel('Number of People')
plt.legend()
plt.show()

lstm_data = table.groupby(['ObservationDate']).agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'})

training_set = lstm_data.iloc[:,0:1].values
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

import numpy as np
X, y = [], []
time_steps = 45
for i in range(len(training_set) - time_steps):
    x = training_set_scaled[i:(i+time_steps), 0]
    X.append(x)
    y.append(training_set_scaled[i+time_steps, 0])
X = np.array(X)
y = np.array(y)
    
split = int(len(X) * 0.8)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
model = Sequential()
model.add(Input(shape=(1, time_steps)))
model.add(LSTM(48, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(48, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(48))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss = 'mean_squared_error',
              optimizer = 'adam',
              metrics = ['mean_squared_error'])
model.summary()
from keras.callbacks import ReduceLROnPlateau
batchsize = 100
epochs =  100
learning_rate_reduction = ReduceLROnPlateau(monitor='val_mean_squared_error', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=1e-10)
history = model.fit(X_train,
                    y_train,
                    batch_size=batchsize,
                    epochs=epochs,
                    validation_split=0.2,
                    shuffle=False,
                    callbacks=[learning_rate_reduction])

y_pred = model.predict(X_test)
y_pred = sc.inverse_transform(y_pred)
y_test = sc.inverse_transform(y_test.reshape(-1,1))
plt.plot(y_pred, color='red')
plt.plot(y_test, color='blue')
plt.title('Actual vs. Predicted Covid Cases (Test Data)')
plt.ylabel('Number of Cases')
plt.xlabel('Day')
plt.legend(['predicted', 'actual'])
