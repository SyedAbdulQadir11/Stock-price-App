import streamlit as st # importing streamlit for web app
import yfinance as yf # importing yfinance for stock data
import pandas as pd # importing pandas for data manipulation
import numpy as np # importing numpy for numerical analysis
import matplotlib.pyplot as plt # importing matplotlib for data visualization
import seaborn as sns # importing seaborn for data visualization
import plotly.graph_objects as go # importing plotly for data visualization
import plotly.express as px # importing plotly for data visualization
import datetime # importing datetime for date and time manipulation
from datetime import date, timedelta # importing datetime for date and time manipulation
from statsmodels.tsa.seasonal import seasonal_decompose # importing seasonal_decompose for time series analysis
import statsmodels.api as sm # importing statsmodels for time series analysis
from statsmodels.tsa.stattools import adfuller # importing adfuller for time series analysis
# title
st.title("""
Stock Market Forecasting and Analysis
""")
st.subheader("""
This web app is developed to forecast and analyze the stock market.
""")
# add an image from online resource
st.image('https://imageio.forbes.com/specials-images/imageserve/617ab453e95e58ee7ce7de16/0x0.jpg?format=jpg&width=1200', width=365)
# take input from user for stock ticker
st.sidebar.header('Select the Parameters')
start_date = st.sidebar.date_input("Start Date", date(2023,1,1))
end_date = st.sidebar.date_input("End Date", date(2023,9,20))
# add ticker symbol list
ticker_list=["AAPL",             
"MSFT",
"GOOG",	
"AMZN",	
"NVDA",	
"TSLA",	
"META",	
"JPM",
"JNJ",
"BTC-USD",
"ETH-USD",
"CSCO",
"BABA",
"MCD",
"NFLX",
"AMD",
"IBM"]
ticker=st.sidebar.selectbox('Select the Company',ticker_list)
# fetch data from yfinance
df=yf.download(ticker,start_date,end_date)
df.insert(0, "Date", df.index, True)
df.reset_index(drop=True,inplace=True)
# display the data
# data from start date to end date
st.write("Data from {} to {}".format(start_date,end_date))
st.write(df)
# plot the data
st.subheader("Data Visualization")
fig=px.line(df,x='Date',y=df.columns,title='Closing Price')
st.plotly_chart(fig)
# add a select box to select columns from data
column=st.selectbox('Select the Column',df.columns[1:])
# subsetting data
df_sub=df[['Date',column]]
st.write('Data for {}'.format(column))
st.write(df_sub)
# ADF test for stationarity
st.subheader("ADF Test for Stationarity")
st.write("If p-value is less than 0.05, then the data is stationary.")
st.write(adfuller(df_sub[column])[1]<0.05)
# lets decompose the data
st.subheader("Decomposition of Time Series")
decomposition = seasonal_decompose(df_sub[column], model='additive', period=7)
fig = decomposition.plot()
st.plotly_chart(fig)
# make same plot in plotly
st.subheader("Decomposition of Time Series in Plotly")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_sub['Date'], y=decomposition.trend, name="Trend",line_color='deepskyblue'))
fig.add_trace(go.Scatter(x=df_sub['Date'], y=decomposition.seasonal,name="Seasonality",line_color='dimgray'))
fig.add_trace(go.Scatter(x=df_sub['Date'], y=decomposition.resid,name="Residual",line_color='red'))
fig.add_trace(go.Scatter(x=df_sub['Date'], y=decomposition.observed,name="Observed",line_color='green'))
fig.update_layout(title='Decomposition of Time Series',xaxis_title='Date',yaxis_title='Closing Price')
st.plotly_chart(fig)
# lets run the model
# user inputs for three parameters p,d,q
p=st.slider('Select the value of p',0,10,1)
d=st.slider('Select the value of d',0,10,1)
q=st.slider('Select the value of q',0,10,1)
seasonal_order=st.number_input('Select the value of seasonal order',0,10,2)
# create model
model=sm.tsa.statespace.SARIMAX(df_sub[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model=model.fit()
# print summary
st.subheader("Model Summary")
st.write(model.summary())
st.write('------')
# forecast the data
st.write('<p style="color:salmon;font-size:25px",font-weight:bold>Forecasting</p>',unsafe_allow_html=True)
forecast_days=st.number_input('Select the number of days for forecast',1,365,30)
prediction=model.get_prediction(start=len(df_sub),end=len(df_sub)+forecast_days-1)
prediction=prediction.predicted_mean
prediction=pd.DataFrame(prediction)
prediction.reset_index(drop=True,inplace=True)
prediction.columns=['Forecast']
st.write(prediction)
# add index to prediction
index=[]
for i in range(len(df_sub),len(df_sub)+forecast_days):
  index.append(i)
prediction['index']=index
# reset index
prediction.set_index('index',inplace=True)
st.write('Prediction for {}'.format(column))
st.write(prediction)
# plot the forecast
st.subheader("Forecast Plot")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub[column], name="Actual",line_color='deepskyblue'))
fig.add_trace(go.Scatter(x=prediction.index, y=prediction['Forecast'],name="Forecast",line_color='red'))
fig.update_layout(title='Forecast Plot',xaxis_title='Date',yaxis_title='Closing Price')
st.plotly_chart(fig)
# plot the residuals
st.subheader("Residual Plot")
residuals = model.resid
fig = px.line(residuals,title='Residual Plot')
st.plotly_chart(fig)
# plot the histogram of residuals
st.subheader("Histogram of Residuals")
fig = px.histogram(residuals,title='Histogram of Residuals')
st.plotly_chart(fig)
# plot the qqplot of residuals
st.subheader("QQPlot of Residuals")
fig = sm.qqplot(residuals,line='s')
st.pyplot(fig)
# add buttons to show and hide separate plots
show_plots=False
if st.button('Show Plots'):
  show_plots=True
if show_plots:
    # plot the acf plot
    st.subheader("ACF Plot")
    fig = sm.graphics.tsa.plot_acf(df_sub[column],lags=20)
    st.pyplot(fig)
    # plot the pacf plot
    st.subheader("PACF Plot")
    fig = sm.graphics.tsa.plot_pacf(df_sub[column],lags=20)
    st.pyplot(fig)
    # plot the seasonal acf plot
    st.subheader("Seasonal ACF Plot")
    fig = sm.graphics.tsa.plot_acf(df_sub[column],lags=20)
    st.pyplot(fig)
    # plot the seasonal pacf plot
    st.subheader("Seasonal PACF Plot")
    fig = sm.graphics.tsa.plot_pacf(df_sub[column],lags=20)
    st.pyplot(fig)
    # button for hiding plots
    if st.button('Hide Plots'):
      show_plots=False
st.write("Developed by: [Syed Abdul Qadir Gilani](https://www.linkedin.com/in/syedabdulqadir/)")
# adding lstm
st.write('<p style="color:salmon;font-size:25px",font-weight:bold>LSTM</p>',unsafe_allow_html=True)
# footer
st.markdown(
    """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: salmon;
    color: white;
    text-align: center;
}
</style>
<div class="footer">
<p>Made with ❤ by Syed Abdul Qadir Gilani</p>
</div>
""",
    unsafe_allow_html=True
)
