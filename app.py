app_py_content = """import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(layout="wide")
st.title('Tractor Sales Analysis and Forecasting')

# --- 1. Data Loading and Preprocessing ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')
    df1 = df.set_index('Month-Year')
    return df, df1

df, df1 = load_data('Tractor-Sales.csv') # Ensure this path is correct for deployment

st.header('Original Tractor Sales Data')
st.write(df)

# --- 2. Moving Averages ---
st.header('Moving Averages of Tractor Sales')

six_month_moving_avg = df1.rolling(6).mean()
nine_month_moving_avg = df1.rolling(9).mean()
twelve_month_moving_avg = df1.rolling(12).mean()

plot_ma_df = pd.DataFrame({
    'Original Sales': df1['Number of Tractor Sold'],
    '6-Month MA': six_month_moving_avg['Number of Tractor Sold'],
    '9-Month MA': nine_month_moving_avg['Number of Tractor Sold'],
    '12-Month MA': twelve_month_moving_avg['Number of Tractor Sold']
})

fig_ma = px.line(
    plot_ma_df,
    x=plot_ma_df.index,
    y=['Original Sales', '6-Month MA', '9-Month MA', '12-Month MA'],
    title='Tractor Sales with Moving Averages',
    labels={'value': 'Number of Tractor Sold', 'Month-Year': 'Date'},
    hover_name=plot_ma_df.index.strftime('%Y-%m-%d')
)
fig_ma.update_layout(hovermode='x unified')
st.plotly_chart(fig_ma, use_container_width=True)


# --- 3. Seasonal Decomposition ---
st.header('Seasonal Decomposition of Tractor Sales')

decomposition = seasonal_decompose(df1['Number of Tractor Sold'], model='additive')

fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=('Original Sales', 'Trend', 'Seasonal', 'Residual'))

fig_decomp.add_trace(go.Scatter(x=df1.index, y=df1['Number of Tractor Sold'], mode='lines', name='Original'), row=1, col=1)
fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)

fig_decomp.update_layout(title_text='Seasonal Decomposition of Tractor Sales (Plotly)', height=1000)
st.plotly_chart(fig_decomp, use_container_width=True)


# --- 4. Exponential Smoothing Forecasting ---
st.header('Tractor Sales Forecasting using Exponential Smoothing (2015)')

model = ExponentialSmoothing(df1['Number of Tractor Sold'],
                             seasonal_periods=12,
                             trend='add',
                             seasonal='add',
                             initialization_method="estimated").fit()

forecast_2015 = model.forecast(12)

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(
    x=df1.index,
    y=df1['Number of Tractor Sold'],
    mode='lines',
    name='Original Sales'
))
fig_forecast.add_trace(go.Scatter(
    x=model.fittedvalues.index,
    y=model.fittedvalues,
    mode='lines',
    name='Fitted Values',
    line=dict(dash='dash')
))
fig_forecast.add_trace(go.Scatter(
    x=forecast_2015.index,
    y=forecast_2015,
    mode='lines',
    name='2015 Forecast',
    line=dict(color='red')
))
fig_forecast.update_layout(
    title_text='Tractor Sales Forecasting using Exponential Smoothing (2015)',
    xaxis_title='Date',
    yaxis_title='Number of Tractor Sold',
    hovermode='x unified',
    height=600
)
st.plotly_chart(fig_forecast, use_container_width=True)


st.subheader('2015 Forecasted Tractor Sales')
st.write(forecast_2015.to_frame(name='Forecasted Sales'))
"""

with open('app.py', 'w') as f:
    f.write(app_py_content)

print("app.py has been created in the current directory. You can download it from the Colab file browser (left-hand sidebar -> 'Files' icon).")
