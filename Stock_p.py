# importing libraries
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from PIL import Image


# Declaring start date and current date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Adding a title,subtitle and cover image
st.title('Stock Prediction Web Application')
st.subheader('By Albert A. Arthur')
image = Image.open("C:/Users/alber/Desktop/FYP/coverImage.png")
st.image(image, use_column_width=True)

# Declaring stock list
stocks = ('TSLA', 'GOOG', 'AAPL', 'MSFT', 'GME', 'ALPP', 'FB', 'TWTR', 'NFLX', 'NIO', 'SNAP')
selected_stock = st.selectbox('Select dataset for prediction ({})'.format(stocks), stocks)

# Setting default value
default_value_goes_here = 'TSLA'
selected_stock = st.text_input("Input stock ticker symbol for prediction e.g. TSLA,GME etc.", default_value_goes_here)

n_years = st.slider('Years ahead to predict prediction:', 1, 3)
period = n_years * 365


# Collecting cache data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


st.spinner(text='Training progress...')

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

add_selectbox = st.sidebar.text_area(
    "Tip:", "--->Type in a stock symbol! \n ---> View stocklist in dropdown menu"
)


# Plotting Historical Data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Historical data ', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Retrieved model training data
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Creating a model for fbprophet and insert the training data
m = Prophet(daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plotting forecast
st.markdown("""
 
 Forecast prediction
""")
st.write(forecast.tail())

st.markdown("### Forecast plot for {} years".format(n_years))
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
st.markdown("""
Forecast seasonal components:
Time series broken apart into seasonal components. Can give you an idea of what happens on a regular basis with the data.
""")
# Plotting forecast components
fig2 = m.plot_components(forecast)
st.write(fig2)
