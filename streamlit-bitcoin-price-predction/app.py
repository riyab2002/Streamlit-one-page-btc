import streamlit as st
from streamlit_option_menu import option_menu

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 1


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["About","Dashboard", "Prediction", "Sentiments","Metrics","Graphs","Tracker"],  # required
                icons=["segmented-nav","house", "currency-exchange", "emoji-smile","book-half","file-bar-graph","question-diamond"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    # if example == 1:
    #     # 2. horizontal menu w/o custom style
    #     selected = option_menu(
    #         menu_title=None,  # required
    #         options=["Home", "Projects", "Contact","dfffgdhdh","dfawhytehgh","rgbfdhr"],  # required
    #         icons=["house", "book", "envelope"],  # optional
    #         menu_icon="cast",  # optional
    #         default_index=0,  # optional
    #         orientation="horizontal",
    #     )
    #     return selected

   
selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Dashboard":
    import streamlit as st
    st.title("Crypto Dashboard")
    import streamlit as st
    import pandas as pd

    st.subheader('**Selected Price**')
    df = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')
    def round_value(input_value):
        if input_value.values > 1:
            a = float(round(input_value, 2))
        else:
            a = float(round(input_value, 8))
        return a

    crpytoList = {
        'Price ': 'BTCBUSD',
     }

    col1, col2, col3 = st.columns(3)

    for i in range(len(crpytoList.keys())):
        selected_crypto_label = list(crpytoList.keys())[i]
        selected_crypto_index = list(df.symbol).index(crpytoList[selected_crypto_label])
        selected_crypto = st.selectbox(selected_crypto_label, df.symbol, selected_crypto_index, key = str(i))
        col_df = df[df.symbol == selected_crypto]
        col_price = round_value(col_df.weightedAvgPrice)
        col_percent = f'{float(col_df.priceChangePercent)}%'
        if i < 3:
            with col1:
                st.metric(selected_crypto, col_price, col_percent)
        if 2 < i < 6:
            with col2:
                st.metric(selected_crypto, col_price, col_percent)
        if i > 5:
            with col3:
                st.metric(selected_crypto, col_price, col_percent)

    st.header('**All Price**')
    st.dataframe(df)

elif selected == "Sentiments":
    import snscrape.modules.twitter as sntwitter
    from tqdm.notebook import tqdm
    import pandas as pd
    import streamlit as st
    import pandas as pd
    import numpy as np
    import re
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('ggplot')
    from textblob import TextBlob
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    from wordcloud import WordCloud
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
     


    st.header("Twitter Sentiment Analysis For Bitcoin")
    day2=st.date_input("From")
    day1=st.date_input("Until")

    q="#Bitcoin (from:DocumentingBTC OR from:MartyBent OR from:lopp OR from:nic__carter OR from:Gladstein OR from:BTC_Archive) until:"+str(day1)+" since:"+str(day2)
    # st.write(q)
    # query="#Bitcoin (from:DocumentingBTC OR from:MartyBent OR from:lopp OR from:nic__carter OR from:Gladstein OR from:BTC_Archive) until:2023-03-25 since:2021-09-26"
    query=q
    scrapper=sntwitter.TwitterSearchScraper(query)
    tweets=[]

     
    for i,tweet in enumerate(scrapper.get_items()):
        data=[
        tweet.date,
        tweet.id, 
        tweet.rawContent,
        tweet.user.username,
        tweet.likeCount,
        tweet.retweetCount,
        ]
        tweets.append(data)
        if i>10:
            break


    df=pd.DataFrame(tweets,columns=["date_str","id","text","username","like_count","retweet_count"])




    text_df = df.drop(["date_str","id","username","like_count","retweet_count"], axis=1)
    # st.write(text_df.head())



    def data_processing(text):
        text = text.lower()
        text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#','',text)
        text = re.sub(r'[^\w\s]','',text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)


    text_df.text = text_df['text'].apply(data_processing)

    text_df = text_df.drop_duplicates('text')




    stemmer = PorterStemmer()
    def stemming(data):
        text = [stemmer.stem(word) for word in data]
        return data

    text_df['text'] = text_df['text'].apply(lambda x: stemming(x))

    text_df.head()

    def polarity(text):
        return TextBlob(text).sentiment.polarity



    text_df['polarity'] = text_df['text'].apply(polarity)


    text_df.head(10)


    def sentiment(label):
        if label <0:
            return "Negative"
        elif label ==0:
            return "Neutral"
        elif label>0:
            return "Positive"



    text_df['sentiment'] = text_df['polarity'].apply(sentiment)


    if st.button("show"):
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.text(" ")

        fig = plt.figure(figsize=(5,5))
        sns.countplot(x='sentiment', data = text_df)
        st.write(fig)


    # fig = plt.figure(figsize=(7,7))
    # colors = ("yellowgreen", "gold", "red")
    # wp = {'linewidth':2, 'edgecolor':"black"}
    # tags = text_df['sentiment'].value_counts()
    # explode = (0.1,0.1,0.1)
    # tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
    #          startangle=90, wedgeprops = wp, explode = explode, label='')
    # plt.title('Distribution of sentiments')
    # st.write(fig)

    pos_tweets = text_df[text_df.sentiment == 'Positive']
    pos_tweets = pos_tweets.sort_values(['polarity'], ascending= False)
    pos_tweets.head()
elif selected == "Metrics":
    # st.title(f"You have selected {selected}")

    import streamlit as st
    import yfinance as yf
    import os
    import pandas as pd
    btc_ticker = yf.Ticker("BTC-USD")
    if os.path.exists("btc.csv"):
        btc = pd.read_csv("btc.csv", index_col=0)
    else:
        btc = btc_ticker.history(period="max")
        btc.to_csv("btc.csv")
    btc.index = pd.to_datetime(btc.index)
    del btc["Dividends"]
    del btc["Stock Splits"]
    df=pd.read_csv("btc.csv")
    choice = option_menu(
        menu_title=None,  # required
        options=["Preview dataset","Show Description"],  # required
        icons=["house", "book"],  # optional
        # menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
    )
    st.text(" ")
    st.text(" ")
    st.text(" ")
    # return selected
    # menu=["Preview dataset","Show Description"]
    # choice=st.sidebar.selectbox("menu",menu)
    if choice=="Preview dataset":
        number=st.number_input("number to show")
        number = int(number) 
        st.dataframe(df.head(number))
        particular=["Rows","Columns"]
        ch=st.selectbox("Select",particular)
        if ch=="Rows":
            all_columns=df.columns.tolist()
            selected_columns=st.multiselect("select Columns",all_columns)
            new_df=df[selected_columns]
            st.dataframe(new_df)

        if ch=="Columns":
            selected_index=st.multiselect("Select Rows",df.head(10).index)
            selected_rows=df.loc[selected_index]
            st.dataframe(selected_rows)

    elif choice=="Show Description":
        st.write(df.describe())
        particular=["Column names","Shape of dataset","Show Dimension","Value Counts"]
        ch=st.selectbox("Description of dataframe",particular)
        if ch=="Column names":
            st.write(df.columns)
    #           #Description
        if ch=="Shape of dataset":
            st.write(df.shape)
        if ch=="Show Dimension":
            data_dim=st.radio("show dimension by",("Rows","Columns"))
            if data_dim=='Rows':
                st.text("Number of Rows")
                st.write(df.shape[0])
            elif data_dim=='Columns':
                st.text("Number of Columns")
                st.write(df.shape[1])
            else:
                    st.write(df.shape)
        if ch=="Value Counts":
            st.text("Value Counts By class")
            st.write(df.iloc[:,-1].value_counts())

elif selected == "Graphs":
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import datetime 
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    from bs4 import BeautifulSoup
    import base64

    def app():
        
        @st.cache_data
        def locknload():
            s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
            for i in range(0,len(s_coins)):
                coin = s_coins[i]
                if i==0:
                    btc = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
                if i==1:
                    eth = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
                if i==2:
                    usdt = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
                if i==3:
                    bnb = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
                if i==4:
                    usdc = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
                if i==5:
                    xrp = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
                if i==6:
                    ada = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
                if i==7:
                    matic = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
                if i==8:
                    doge = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
                if i==9:
                    busd = yf.download(tickers=coin,start="2023-01-01",end=datetime.datetime.now())
                    continue
            return btc,eth,usdt,bnb,usdc,xrp,ada,matic,doge,busd
        
        #page layout
        # st.set_page_config(layout="wide")
        # mainpage_bg = '''<style>
        # [data-testid="stAppViewContainer"]>.main{{
        # background-image:url("image/img_file.jpg");
        # background-size : cover;
        # background-position : top left;
        # background-repeat : no-repeat;
        # backgorund-attachment:local;}}
        # [data-testid="stHeader"]
        # {{background:rgba(0,0,0,0);
        # }}
        # [data-testid="stToolbar"]
        # {{right: 2rem;}}
        # </style>'''
        # st.markdown(mainpage_bg,unsafe_allow_html=True)
        #Title
        st.title("Crypto Prediction")

        #About at last
        col1=st.sidebar
        col1.header("Crypto Coins")

        currency_unit = "USD"

        s_coins = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','MATIC-USD','DOGE-USD','BUSD-USD']
        coin = 'BTC-USD'

        btc,eth,usdt,bnb,usdc,xrp,ada,matic,doge,busd=locknload()

        st.subheader(" Price Data of "+coin)
        
        
        if coin =='BTC-USD':
            df=btc.copy()
        if coin =='ETH-USD':
            df=eth.copy()
        if coin =='USDT-USD':
            df=usdt.copy()
        if coin =='BNB-USD':
            df=bnb.copy()
        if coin =='USDC-USD':
            df=usdc.copy()
        if coin =='XRP-USD':
            df=xrp.copy()
        if coin =='ADA-USD':
            df=ada.copy()
        if coin =='MATIC-USD':
            df=matic.copy()
        if coin =='DOGE-USD':
            df=doge.copy()
        if coin =='BUSD-USD':
            df=busd.copy()
        
        # st.table(df.tail(5))

        def download_file(df):
            csv=df.to_csv(index=True)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
            return href
        st.markdown(download_file(df),unsafe_allow_html=True)

        figure=go.Figure()

        figure.add_trace(go.Scatter(x=df.index,y=df['Close'],name='Close'))

        figure.add_trace(go.Scatter(x=df.index,y=df['Open'],name='Open'))

        figure.update_layout(title='Opening and Closing Prices',yaxis_title='Crypto Price (USD)',height=700,width=1300)

        st.subheader('Opening and Closing Prices Of '+str(coin))
        
        st.plotly_chart(figure)
        #Low and high
        figure_LH=go.Figure()

        figure_LH.add_trace(go.Scatter(x=df.index,y=df['Low'],name='Low'))
        figure_LH.add_trace(go.Scatter(x=df.index,y=df['High'],name='High'))

        figure_LH.update_layout(title='High and Low Prices',yaxis_title='Crypto Price (USD)',height=700,width=1300)

        st.subheader('High and Low Prices Of '+str(coin))
        
        st.plotly_chart(figure_LH)

        #yayyy

        figure_vol=go.Figure()

        figure_vol.add_trace(go.Scatter(x=df.index,y=df['Volume'],name='Volume'))

        figure_vol.update_layout(title='Volume Sold',yaxis_title='Crypto Price (USD)',height=700,width=1300)

        st.subheader('Volume of '+str(coin)+' sold')
        st.plotly_chart(figure_vol)

        #st.markdown("""<iframe title="Crytocurrency y updated" width="930" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=b1ae4dd0-0031-49d4-8939-3fd411b967d5&autoAuth=true&ctid=83f3dabc-6f25-4958-bcab-9c1cd2b7fb2e" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)

    app()
elif selected == "Tracker":
    import yfinance as yf
    import streamlit as st
    import pandas as pd
    from datetime import datetime
    from datetime import date
    from dateutil.relativedelta import relativedelta
    import plotly.express as px

    crypto_mapping = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}

    st.title("Bitcoin Tracker")
    menu=["Bitcoin"]
    # crypto_option = st.sidebar.selectbox(
    #     "Which Crypto do you want to visualize?", menu
    # )
    crypto_option = "Bitcoin"
    start_date = st.date_input("Start Date", date.today() - relativedelta(months=1))
    end_date = st.date_input("End Date", date.today())

    data_interval = st.selectbox(
        "Data Interval",
        (
            
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ),
    )

    symbol_crypto = crypto_mapping[crypto_option]
    data_crypto = yf.Ticker(symbol_crypto)

    value_selector = st.selectbox(
        "Value Selector", ("Open", "High", "Low", "Close", "Volume")
    )

    if st.button("Generate"):
        crypto_hist = data_crypto.history(
            start=start_date, end=end_date, interval=data_interval
        )
        fig = px.line(crypto_hist, 
        x=crypto_hist.index, y=value_selector,
        labels={"x": "Date"})
        st.plotly_chart(fig)


elif selected == "Prediction":
    import sqlite3
    import pandas as pd
    import requests
    import os
    import csv
    import json
    #import config


    #Fear and Greed index in 1D timeframe
    CSV_URL = 'https://api.alternative.me/fng/?limit=0&format=csv'


    with requests.Session() as s:
        download = s.get(CSV_URL)

        decoded_content = download.content.decode('utf-8')

        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        my_list = my_list[4:-5]

            
    df = pd.DataFrame(my_list,columns = ['date','fng_value','fng_class'])
    df['date'] = pd.to_datetime(df['date'],format = '%d-%m-%Y')


    #BTC price data in 1D timeframe
    btc = 'https://api.cryptowat.ch/markets/binance/btcusdt/ohlc?periods=86400'
    with requests.Session() as s:
        download = s.get(btc)

        decoded_content = download.content.decode('utf-8')
        
    jn = json.loads(decoded_content)
    price = pd.DataFrame(jn['result']['86400'], columns = ['date','Open','High', 'Low', 'Close', 'Volume_btc','Volume_usd'])
    price['date'] = pd.to_datetime(price['date'],unit = 's')    


    #SOPR value in 1D timeframe
    onc = requests.get('https://api.cryptoquant.com/v1/Bitcoin/Exchange-Flows')
    #API_KEY = config.api_key
    API_KEY = '277MmKOdUSzI12ciSkp9xzyqrOr'


    res = requests.get('https://api.glassnode.com/v1/metrics/indicators/sopr',
        params={'a': 'BTC', 'api_key': API_KEY})

    rs = res.content.decode('utf-8')

    # convert to pandas dataframe
    sopr = pd.DataFrame(res.json())
    sopr['t'] = pd.to_datetime(sopr['t'], unit='s')
    sopr.rename(columns={'t':'date', 'v':'sopr_val'}, inplace = True)


    # Join the three tables on date into a single dataframe

    pfng = df.merge(price, on = 'date', how='inner')
    pfngs = pfng.merge(sopr, on = 'date', how='inner')
    pfngs = pfngs[::-1].reset_index(drop = True)
    #pfngs.to_csv(r'btc_pfngs.csv', index=False)
    pfngs.set_index('date', inplace=True)

    # Drop fear-Greed categorical variable as fear and greed value is retained
    pfngs.drop(columns='fng_class', inplace=True)

    conn = sqlite3.connect("btc.db")
    cursor = conn.cursor()

    pfngs.to_sql('btcprice',conn,if_exists="replace")

    conn.close()


    import streamlit as st
    import pandas as pd
    import sqlite3
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    import mpld3
    import streamlit.components.v1 as components
    from datetime import timedelta, date
    import plost

    @st.cache(allow_output_mutation=True)
    def load_data(db):
        conn = sqlite3.connect(db)
        df = pd.read_sql_query("SELECT * FROM btcprice;", conn)
        conn.close()
        df['date'] = pd.to_datetime(df['date'])
        return df

    def load_model(model,weight):
        model_uni = keras.models.load_model(model)
        model_uni.load_weights(weight)
        return model_uni


    def create_test_next(test_X, model, window_size = 75, future_days = 30):
        new_test_X = test_X
        for i in range(future_days):
            Last = new_test_X[-1:]
            last2 = Last[0,1:]
            new = np.append(last2,model.predict(new_test_X[-1:]),axis=0)
            #print(new)
            new_test_X = np.append(new_test_X,new.reshape(1,window_size,1),axis=0)
        return new_test_X

    df = load_data('btc.db')
    model_uni = load_model(model='Univariate_model', weight='uni_weights.h5')

    scaler = MinMaxScaler(feature_range=(0,1))
    close = np.array(df.iloc[:,5]).reshape(-1,1)
    scaler.fit(close)

    last75 = np.array(df.iloc[-75:,5]).reshape(-1,1)
    last75_sc = scaler.transform(last75)
    test_last = np.reshape(last75_sc,(1,75,1))
    next_test5 = create_test_next(test_last, model_uni,window_size = 75,future_days = 5)
    next_price5 = model_uni.predict(next_test5)
    next_5_price = scaler.inverse_transform(next_price5)

    st.title("Bitcoin Price Prediction")

    st.header("Next 5 days Close Price")

    from datetime import date
    today = date.today()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label = "Today's Close", value ="{price1} $".format(price1=round(next_5_price[0][0])) , delta ="{:.2f} %".format((next_5_price[0][0] - last75[-1][0]) /last75[-1][0]*100 ))
    col2.metric(label = "Tomorrow's Close", value ="{price1} $".format(price1=round(next_5_price[1][0])) , delta ="{:.2f} %".format((next_5_price[1][0] - next_5_price[0][0])/next_5_price[0][0]*100 ))
    col3.metric(label = today.strftime('%B')+ ' '+str(today.day+2), value ="{price1} $".format(price1=round(next_5_price[2][0])) , delta ="{:.2f} %".format((next_5_price[2][0] - next_5_price[1][0])/next_5_price[1][0]*100 ))
    col4.metric(label = today.strftime('%B')+ ' '+str(today.day+3), value ="{price1} $".format(price1=round(next_5_price[3][0])) , delta ="{:.2f} %".format((next_5_price[3][0] - next_5_price[2][0])/next_5_price[2][0]*100 ))
    col5.metric(label = today.strftime('%B')+ ' '+str(today.day+4), value ="{price1} $".format(price1=round(next_5_price[4][0])) , delta ="{:.2f} %".format((next_5_price[4][0] - next_5_price[3][0])/next_5_price[3][0]*100 ))

    next_d = []
    for i in range(5):
        #next_d = []
        x = (date.today() + timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S')
        next_d.append(x)
    next_days = pd.to_datetime(next_d)

    tab1, tab2 = st.tabs(["Historic","Prediction"])

    with tab1:
        st.header('Historic Price')
        plost.line_chart(data=df, x='date', y='Close', height=500, width=600, pan_zoom='minimap', use_container_width=True)

    with tab2:
        st.header('Prediction for next 5 days')
        dates = pd.to_datetime(df['date'])
        fig = plt.figure()
        sns.lineplot(y=df.iloc[-90:,5],x=dates[-90:])
        plt.xlabel('Date')
        plt.ylabel('BTC Price in USD')
        sns.lineplot(x=next_days,y=next_5_price[0:5].reshape(-1))
        plt.legend(['Historic','Predicted'])
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)

elif selected == "About":

    import streamlit as st

    # st.set_page_config(
    #     page_title="Bitcoin Price Prediction",
    #     page_icon="👋",
    # )
    import json
    import time

    import requests
    import streamlit as st
    from streamlit_lottie import st_lottie
    from streamlit_lottie import st_lottie_spinner


    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)


    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    st.title("Predict Bitcoin Price")   
    lottie_streamlit = load_lottiefile("./lottiefiles/Streamlit Logo Animation.json")
    lottie_progress = load_lottiefile("./lottiefiles/44327-animated-rocket-icon.json")
    lottie_success = load_lottiefile("./lottiefiles/26514-check-success-animation.json")
    lottie_error = load_lottiefile("./lottiefiles/38463-error.json")
    lottie_url="https://assets3.lottiefiles.com/packages/lf20_yc9ywdm7.json"
    # )
    downloaded_url = load_lottieurl(lottie_url)

    if downloaded_url is None:
        col1, col2 = st.columns((2, 1))
        col1.warning(f"URL {lottie_url} does not seem like a valid lottie JSON file")
        with col2:
            st_lottie(lottie_error, height=100, key="error")
    else:
        # with st.echo("above"):
            st_lottie(downloaded_url, key="user")

    # txt = st.text_area('Text to analyze', '''
    # It was the best of times, it was the worst of times, it was
    # the age of wisdom, it was the age of foolishness, it was
    # the epoch of belief, it was the epoch of incredulity, it
    # was the season of Light, it was the season of Darkness, it
    # was the spring of hope, it was the winter of despair, (...)
    # ''',heigh)
    # st.write('Sentiment:', run_sentiment_analysis(txt))


    # st.title("Main Page")
    # st.sidebar.success("Select a page above.")

    # if "my_input" not in st.session_state:
    #     st.session_state["my_input"] = ""

    # my_input = st.text_input("Input a text here", st.session_state["my_input"])
    # submit = st.button("Submit")
    # if submit:
    #     st.session_state["my_input"] = my_input
    #     st.write("You have entered: ", my_input)
    # st.markdown()
















