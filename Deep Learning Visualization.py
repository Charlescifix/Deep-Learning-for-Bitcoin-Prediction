#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.figure_factory as ff


# In[2]:


token = pd.read_csv("BTC_GBP_hourly.csv")


# In[3]:


token.head(5)


# In[4]:


token = token.drop(columns = ['conversionType', 'conversionSymbol'])


# In[5]:


token['time'] = pd.to_datetime(token['time'])
token = token.set_index('time')
token = token.sort_index()


# In[6]:


token.head()


# In[ ]:





# In[7]:


# Simple Moving Average
sma_period = 24  # One full day


# Relative Strength Index
rsi_period = 24  # Commonly used parameter


# Stochastic Oscillator
stoch_period = 24  # Commonly used parameter


# Moving Average Convergence Divergence
macd_fast_period = 12  # Adjusted from daily data
macd_slow_period = 24  # Adjusted from daily data
macd_signal_period = 12  # Commonly used parameter


# Exponential Moving Average
ema_short_period = 12  # Short term EMA
ema_long_period = 24  # Long term EMA


# Average True Range
atr_period = 24  # Commonly used parameter


# Bollinger Bands
bb_period = 24  # Commonly used parameter
bb_std_dev = 2  # Commonly used parameter


# In[8]:


token['volatility'] = (token['high'] - token['low']) / token['low'] * 100

token['hourly_return'] = (token['close'] - token['open']) / token['open'] * 100

token['avg_price'] = (token['high'] + token['low'] + token['close']) / 3

token['relative_volume'] = token['volumefrom'] / token['volumefrom'].rolling(window=20).mean()  

token['momentum'] = token['close'] - token['close'].shift(1) 

# Simple Moving Average

token['sma'] = token['close'].rolling(window=sma_period).mean()


# Relative Strength Index

delta = token['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
rs = gain / loss
token['rsi'] = 100 - (100 / (1 + rs))

# Stochastic Oscillator

low_min = token['low'].rolling(window=stoch_period).min()
high_max = token['high'].rolling(window=stoch_period).max()
token['stoch'] = 100 * (token['close'] - low_min) / (high_max - low_min)

# Moving Average Convergence Divergence (MACD)

token['ema12'] = token['close'].ewm(span=macd_fast_period, adjust=False).mean()
token['ema24'] = token['close'].ewm(span=macd_slow_period, adjust=False).mean()
token['macd'] = token['ema12'] - token['ema24']
token['macd_signal'] = token['macd'].ewm(span=macd_signal_period, adjust=False).mean()
token['macd_hist'] = token['macd'] - token['macd_signal']

# Exponential Moving Average (EMA)

token['ema_short'] = token['close'].ewm(span=ema_short_period, adjust=False).mean()
token['ema_long'] = token['close'].ewm(span=ema_long_period, adjust=False).mean()

# Average True Range (ATR)

high_low = token['high'] - token['low']
high_close = np.abs(token['high'] - token['close'].shift())
low_close = np.abs(token['low'] - token['close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
token['atr'] = true_range.rolling(window=atr_period).mean()

token['sma20'] = token['close'].rolling(window=bb_period).mean()
token['std20'] = token['close'].rolling(window=bb_period).std()
token['upper_band'] = token['sma20'] + (bb_std_dev * token['std20'])
token['lower_band'] = token['sma20'] - (bb_std_dev * token['std20'])

# Define the columns for which you want to create lagged features
columns_to_lag = ['close', 'volumeto']

# Define the number of lagged hours
lagged_hours = [24]

# Create lagged features for each column and each number of hours
for column in columns_to_lag:
    for hour in lagged_hours:
        token[f'{column}_lagged_{hour}_hour'] = token[column].shift(hour)


# In[ ]:





# In[9]:


token.head()


# In[10]:


token.dropna(inplace = True)


# In[11]:


token.drop_duplicates()


# In[ ]:





# In[13]:


# Creating a larger figure for better readability
fig, ax = plt.subplots(figsize=(20, 15))

# Generating the correlation matrix
correlation_matrix = token.corr()

# Creating a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 8}, fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})

# Adding titles and labels for clarity
plt.title('Correlation Matrix of Features', fontsize=20)
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10)

# Showing the plot
plt.show()


# In[ ]:





# In[14]:


fig = px.line(token, y='volatility', title='Bitcoin Price Volatility')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Volatility (%)')
fig.show()


# In[ ]:





# In[15]:


# SMA and Close Price
fig = px.line(token, y=['close', 'sma'], title='SMA and Close Price')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Price')
fig.show()

# MACD and Signal Line
fig = px.line(token, y=['macd', 'macd_signal'], title='MACD and Signal Line')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Value')
fig.show()

# RSI
fig = px.line(token, y='rsi', title='Relative Strength Index (RSI)')
fig.add_hline(y=70, line_dash="dash", line_color="red")
fig.add_hline(y=30, line_dash="dash", line_color="green")
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='RSI')
fig.show()


# In[16]:


# Selecting a broader range of features for correlation analysis
features_for_correlation = ['close', 'volumefrom', 'volatility', 'sma', 'rsi', 
                            'hourly_return', 'avg_price', 'momentum', 'stoch', 
                            'macd', 'atr', 'upper_band', 'lower_band']

correlation_matrix = token[features_for_correlation].corr()

fig = ff.create_annotated_heatmap(
    z=correlation_matrix.to_numpy(), 
    x=correlation_matrix.columns.tolist(), 
    y=correlation_matrix.columns.tolist(),
    annotation_text=correlation_matrix.round(2).to_numpy(),
    colorscale='Viridis'
)
fig.update_layout(title_text='Correlation Analysis', title_x=0.5)
fig.show()


# In[ ]:





# In[ ]:





# In[17]:


fig = px.line(token, y='close', title='Bitcoin Price Trend')

# Customizing the x-axis to show more dates
fig.update_xaxes(
    title_text='Date',
    tickmode='auto',      
    nticks=20             
)

fig.update_yaxes(title_text='Close Price')
fig.show()


# In[ ]:





# In[18]:


rolling_correlation = token['close'].rolling(window=30).corr(token['sma'])
fig = px.line(rolling_correlation, title='Rolling Correlation: Close Price and SMA')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Rolling Correlation')
fig.show()


# In[ ]:





# In[19]:


import plotly.express as px

# Creating a histogram for hourly_return
fig = px.histogram(token, x='hourly_return', nbins=100, marginal='box', title='Distribution of Hourly Returns')

# Update layout for readability
fig.update_layout(xaxis_title='Hourly Return (%)', yaxis_title='Frequency', bargap=0.1)

# Showing the plot
fig.show()


# In[ ]:





# In[20]:


import plotly.graph_objects as go

# Scatter plot for 'close' vs 'close_lagged_24_hour'
fig = go.Figure()
fig.add_trace(go.Scatter(x=token['close_lagged_24_hour'], y=token['close'], mode='markers', 
                         name='Close vs Lagged Close'))

fig.update_layout(title='Current Close Price vs 24-hour Lagged Close Price',
                  xaxis_title='24-hour Lagged Close Price',
                  yaxis_title='Current Close Price')

fig.show()

# Scatter plot for 'volumeto' vs 'volumeto_lagged_24_hour'
fig = go.Figure()
fig.add_trace(go.Scatter(x=token['volumeto_lagged_24_hour'], y=token['volumeto'], mode='markers', 
                         name='Volume To vs Lagged Volume To'))

fig.update_layout(title='Current Volume To vs 24-hour Lagged Volume To',
                  xaxis_title='24-hour Lagged Volume To',
                  yaxis_title='Current Volume To')

fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




