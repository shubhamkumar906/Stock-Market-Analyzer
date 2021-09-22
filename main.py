from flask import Flask, redirect, url_for, render_template, request




############## STOCK PRICE PREDICTION StockMarketSimulator Part-1
#########################################
def price_prediction(stock_name,forecast_out):
    import quandl
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split

    df = quandl.get("WIKI/"+stock_name)
    #print(df.head())
    df = df[['Adj. Close']]
    #print(df.head())
    # A variable for predicting 'n' days out into the future
    forecast_out = int(forecast_out)
    # Create another column (the target or dependent variable) shifted 'n' units up
    df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
    #print(df.head())
    #print(df.tail())
    ### Creating the independent dataset(X)
    # Converitng the dataframe to a numpy array
    x = np.array(df.drop(['Prediction'],1))
    # Removing the last 'n' rows
    x = x[:-forecast_out]
    #print(x)
    ### Creating the dependent dataset (Y)
    # Converting the dataframe into a numpy array (All the values including the NaN's)
    y = np.array(df['Prediction'])
    # Get all of the y values except the last 'n' rows
    y = y[:-forecast_out]
    #print(y)
    # Splitting the dataset into 80% training and 20% testing
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    # Creating and training the model i.e. Support Vector Machine (Regressor)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train,y_train)
    # Testing Model : Score returns the coefficient of determination R^2 of the prediction.
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)
    #print('svm_confidence:',svm_confidence)
    # Create and train the Linear Regression Model
    lr = LinearRegression()
    # Training the model
    lr.fit(x_train, y_train)
    # Testing Model : Score returns the coefficient of determination R^2 of the prediction.
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)
    #print('lr_confidence:',lr_confidence)
    # Set a variable x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    #print(x_forecast)
    # Print the Linear Regression Model predictions for next 'n' days
    lr_prediction = lr.predict(x_forecast)
    #print(lr_prediction)
    # Print the Support Vector Model predictions for next 'n' days
    svm_prediction = svr_rbf.predict(x_forecast)
    #print(svm_prediction)
    r = svm_prediction
    return (r)

############## PRICE PREDICTION ends
##############################################




############## STOCK PREDICTION, SIMPLE VOLUME ANALYSIS StockMarketSimulator Part-2
#########################################


"""import time
import yfinance as yf
import pandas as pd
df = pd.read_csv('companylist.csv')
#print(df['Symbol'])
increased_symbols = []
t_end = time.time() + 30

for stock in df['Symbol']:
    while time.time() < t_end:
        stock = stock.upper()
        if '^' in stock:
            pass
        else:
            try:
              stock_info = yf.Ticker(stock)
              hist = stock_info.history(period='5d')
              previous_averaged_volume = hist['Volume'].iloc[1:4:1].mean()
              todays_volume = hist['Volume'][-1]
              if todays_volume > previous_averaged_volume * 4:
                increased_symbols.append(stock)
            except:
              pass
            print(increased_symbols)"""

############## STOCK PREDICTION, SIMPLE VOLUME ANALYSIS ends
##############################################



############## ALERTS AND NOTIFICATIONS, ALERTS BASED ON THRESHOLDS StockMarketSimulator Part-3
#########################################
def stock_alert(symbol,thres):
    import pandas as pd
    from alpha_vantage.timeseries import TimeSeries
    import time

    api_key = '67O5YS2MXMMVBHVL'

    ts = TimeSeries(key=api_key, output_format='pandas')

    data, meta_data = ts.get_intraday(symbol, interval='1min', outputsize='full')
    #print(data)

    close_data = data['4. close']
    percentage_change = close_data.pct_change()

    #print(percentage_change)

    last_change = percentage_change[-1]
    if abs(last_change) > float(thres):
        alt = symbol + " STOCK ALERT !!!    " + str(last_change)
        # print('Microsoft Alert:' + str(last_change))
        return alt
    else:
        return("No alerts for this stock, check after sometime")
############## ALERTS AND NOTIFICATIONS, ALERTS BASED ON THRESHOLDS ends
##############################################



############## COMPARISON_STOCKS, SIMPLE MOVING AVERAGE StockMarketSimulator Part-4
#########################################
def stock_comparison(symbol,interval):
    import pandas as pd
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.techindicators import TechIndicators
    import matplotlib.pyplot as plt
    api_key = '67O5YS2MXMMVBHVL'

    ts = TimeSeries(key = api_key, output_format='pandas')
    data_ts, meta_data = ts.get_intraday(symbol, interval, outputsize = 'full')
    #print(data_ts)

    period = 60
    ti = TechIndicators(key=api_key, output_format='pandas')
    data_ti, meta_data_ti = ti.get_sma(symbol, interval, time_period=period, series_type='close')
    #print(data_ti)

    df1 = data_ti
    df2 = data_ts['4. close'].iloc[period-1::]
    df2.index = df1.index
    total_df = pd.concat([df1,df2], axis=1)
    #print(total_df)
    total_df.plot()
    plt.show()
    r1 = total_df
    return(r1)

############## COMPARISON_STOCKS, SIMPLE MOVING AVERAGE ends
##############################################

############## TECHINCAL INDICATORS StockMarketSimulator Part-5
#########################################
def technical_indicators(symbol,interval):
    import pandas as pd
    from alpha_vantage.techindicators import TechIndicators
    import matplotlib.pyplot as plt
    api_key = '67O5YS2MXMMVBHVL'

    period = 60
    ti = TechIndicators(key=api_key, output_format='pandas')
    data_rsi, meta_data_rsi = ti.get_rsi(symbol, interval, time_period=period, series_type='close')
    data_sma, meta_data_sma = ti.get_sma(symbol, interval, time_period=period, series_type='close')
    #print(data_ti)

    df1 = data_sma.iloc[1::]
    df2 = data_rsi
    df1.index = df2.index

    fig, ax1 = plt.subplots()
    ax1.plot(df1, 'b-')
    ax2 = ax1.twinx()
    ax2.plot(df2, 'r.')
    plt.title("SMA & RSI graph")
    # plt.show()
    r2 = df1,df2
    return (r2)
############## TECHNICAL INDICATORS ends
##############################################


############## TECHNICAL CHARTS
#########################################
def technical_charts(symbol):
    import time
    import datetime
    import pylab
    import matplotlib
    import mpl_finance
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    from mpl_finance import candlestick_ohlc
    import mpld3

    matplotlib.rcParams.update({"font.size": 9})
    stocksToPull = ["ADANIPORTS", "VEDL", "BHARTIARTL", "ZEEL", "TECHM", "SUNPHARMA", "WIPRO", "CIPLA", "HCLTECH",
                    "POWERGRID", "TCS", "RELIANCE", "BPCL", "INFRATEL", "ITC", "HEROMOTOCO", "MARUTI", "GRASIM",
                    "BAJAJ-AUTO", "UPL", "DRREDDY", "TATASTEEL", "GAIL", "INFY", "HINDALCO", "NTPC", "ADANIPORTS",
                    "TATAMOTORS", "HINDUNILVR", "SBIN", "JSWSTEEL", "COALINDIA", "SHREECEM", "NESTLEIND", "TITAN",
                    "EICHERMOT", "BRITANNIA", "ONGC", "M&M", "IOC", "ULTRACEMCO", "BAJAJFINSV", "ICICIBANK", "HDFCBANK",
                    "LT", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "HDFC", "BAJFINANCE", "ASIANPAINT"]


    def rsiFunc(prices, n=14):
        deltas = np.diff(prices)
        seed = deltas[:n + 1]
        up = sum(seed[seed >= 0]) / n
        down = sum(-seed[seed < 0]) / n
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100. / (1 + rs)
        # print(rsi)

        for i in range(n, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta

            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n

            rs = up / down
            rsi[i] = 100. - 100. / (1 + rs)

        # print(rsi)
        return rsi


    def movingAverage(values, window):
        weights = np.repeat(1.0, window) / window
        smas = np.convolve(values, weights, "valid")
        return smas


    def expMovingAverage(values, window):
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= sum(weights)
        a = np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
        return a


    def computeMACD(x, slow=26, fast=12):
        """
        macd line = 12ema - 26ema
        signal line = 9ema of macd line
        histogram = macd line - signal line
        """
        emaSlow = expMovingAverage(x, slow)
        emaFast = expMovingAverage(x, fast)
        return emaSlow, emaFast, emaFast - emaSlow


    def graphData(stock, MA1, MA2):
        # stockFile = "D:\\software_dev\\algorithmicTrading\\stockData\\oneYearOHLC\\" + stock + ".txt"
        stockFile = "oneYearOHLC//" + stock + '.txt'
        print("stock --> ", stock)

        # fetching data from file starts here*****/
        # date, open, high, low, close adj, volume
        date = []
        saveFile = open(stockFile, "r")
        for eachLine in saveFile:
            lineAppend = datetime.datetime.strptime(eachLine.split(",")[0], "%Y-%m-%d")
            date.append(lineAppend)
        saveFile.close()

        openp = []
        saveFile = open(stockFile, "r")
        for eachLine in saveFile:
            openp.append(float(eachLine.split(",")[1]))
        saveFile.close()

        highp = []
        saveFile = open(stockFile, "r")
        for eachLine in saveFile:
            highp.append(float(eachLine.split(",")[2]))
        saveFile.close()

        lowp = []
        saveFile = open(stockFile, "r")
        for eachLine in saveFile:
            lowp.append(float(eachLine.split(",")[3]))
        saveFile.close()

        closep = []
        saveFile = open(stockFile, "r")
        for eachLine in saveFile:
            closep.append(float(eachLine.split(",")[4]))
        saveFile.close()

        volume = []
        saveFile = open(stockFile, "r")
        for eachLine in saveFile:
            volume.append(float(eachLine.split(",")[5]))
        saveFile.close()
        # ______fetching data from file completed_______

        # ....Candlestick data manipulation.....
        x = 0
        y = len(date)
        candleAr = []
        while x < y:
            appendLine = [mdates.date2num(date[x]), float(openp[x]), float(highp[x]), float(lowp[x]), float(closep[x])]
            candleAr.append(appendLine)
            x = x + 1

        # _____Calculating SMA______
        Av1 = movingAverage(closep, MA1)
        Av2 = movingAverage(closep, MA2)
        SP = len(date[MA2 - 1:])

        label1 = str(MA1) + ' SMA'
        label2 = str(MA2) + ' SMA'

        # .......Plotting Candlesticks Chart........
        fig = plt.figure(facecolor="#07000d")
        ax1 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4, facecolor="#07000d")
        candlestick_ohlc(ax1, candleAr[-SP:], width=1, colorup="g", colordown="r")

        # _____Plotting SMA______
        ax1.plot(date[-SP:], Av1[-SP:], "#5998ff", label=label1, linewidth=1.5)
        ax1.plot(date[-SP:], Av2[-SP:], "#e1edf9", label=label2, linewidth=1.5)

        # ___Designing Graphs____
        ax1.tick_params(axis="y", colors="w")
        ax1.tick_params(axis="x", colors="w")
        plt.ylabel("Stock Price")
        ax1.yaxis.label.set_color("w")
        ax1.spines["bottom"].set_color("#599Bff")
        ax1.spines["top"].set_color("#599Bff")
        ax1.spines["left"].set_color("#599Bff")
        ax1.spines["right"].set_color("#599Bff")
        ax1.grid(True, color="w", alpha=0.5)
        maLeg = plt.legend(loc=9, ncol=2, prop={"size": 5.7}, fancybox=True)
        maLeg.get_frame().set_alpha(0.4)
        textEd = pylab.gca().get_legend().get_texts()
        pylab.setp(textEd[0:5], color="w")

        # .....Merging Volume Chart............
        ax1v = ax1.twinx()
        volumeMin = 0
        ax1v.fill_between(date[-SP:], volumeMin, volume[-SP:], facecolor="#00ffe8", alpha=.5)
        ax1v.axes.yaxis.set_ticklabels([])
        ax1v.set_ylim(0, 2 * max(volume))
        ax1v.tick_params(axis="x", colors="w")
        ax1v.tick_params(axis="y", colors="w")
        ax1v.yaxis.label.set_color("w")
        ax1v.spines["bottom"].set_color("#599Bff")
        ax1v.spines["top"].set_color("#599Bff")
        ax1v.spines["left"].set_color("#599Bff")
        ax1v.spines["right"].set_color("#599Bff")

        # Rotating labels on X-axis
        for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)

        # .........Plotting RSI..........
        ax0 = plt.subplot2grid((6, 4), (0, 0), rowspan=1, colspan=4, facecolor="#07000d", sharex=ax1)
        rsi = rsiFunc(closep)
        rsiCol = "#1a8782"
        posCol = "#386d13"
        negCol = "#8f2020"
        ax0.plot(date[-SP:], rsi[-SP:], "#00ffe8", linewidth=1.5)
        ax0.axhline(70, color=negCol)
        ax0.axhline(30, color=posCol)
        ax0.fill_between(date[-SP:], rsi[-SP:], 70, where=(rsi[-SP:] >= 70), facecolor=rsiCol, edgecolor=rsiCol)
        ax0.fill_between(date[-SP:], rsi[-SP:], 30, where=(rsi[-SP:] <= 30), facecolor=rsiCol, edgecolor=rsiCol)
        # ax0.text(0.015, 0.95, 'RSI (14)', va='top', color='w', transform=ax0.transAxes)
        ax0.set_yticks([30, 70])
        ax0.yaxis.label.set_color("w")
        ax0.tick_params(axis="y", colors="w")
        ax0.yaxis.label.set_color("w")
        ax0.spines["bottom"].set_color("#599Bff")
        ax0.spines["top"].set_color("#599Bff")
        ax0.spines["left"].set_color("#599Bff")
        ax0.spines["right"].set_color("#599Bff")
        plt.ylabel("RSI")

        # .....MACD Plotting......
        ax2 = plt.subplot2grid((6, 4), (5, 0), sharex=ax1, rowspan=1, colspan=4, facecolor="#07000d")
        nslow = 26
        afast = 12
        nema = 9
        fillcolor = "#599Bff"

        emaSlow, emaFast, macd = computeMACD(closep)
        ema9 = expMovingAverage(macd, nema)

        ax2.plot(date[-SP:], macd[-SP:], color="#4ee6fd", lw=2)
        ax2.plot(date[-SP:], ema9[-SP:], color="#e1edf9", lw=1)
        ax2.fill_between(date[-SP:], 0, 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
        ax2.spines["bottom"].set_color("#599Bff")
        ax2.spines["top"].set_color("#599Bff")
        ax2.spines["left"].set_color("#599Bff")
        ax2.spines["right"].set_color("#599Bff")
        ax2.yaxis.label.set_color("w")
        ax2.tick_params(axis="y", colors="w")
        ax2.tick_params(axis="x", colors="w")
        ax2.yaxis.label.set_color("w")
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="upper"))
        plt.ylabel("MACD")

        # Rotating labels on X-axis
        for label in ax2.xaxis.get_ticklabels():
            label.set_rotation(45)

            # END stuff


        plt.suptitle(stock, color="w")
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=0)
        plt.show()
        return


    graphData(symbol, 12, 26)
    return 0

# macd[-SP:]-ema9[-SP:]

##############TECHNICAL CHART ends
##############################################





app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/price_prediction.html", methods=["POST", "GET"])
def Stock_Price_Prediction():
    if(request.method == "POST"):
        stock_name = request.form["stock_name"]
        forecast_out = request.form["forecast_out"]
        ans = price_prediction(stock_name,forecast_out)
        return redirect(url_for("user", usr = ans))
    return render_template("price_prediction.html")

@app.route("/stock_prediction.html", methods=["POST", "GET"])
def Stock_Name_Prediction():
    if(request.method == "POST"):
        ans = stock_prediction()
        return redirect(url_for("user", usr = ans))
    return render_template("stock_prediction.html")


@app.route("/alerts.html", methods=["POST", "GET"])
def AlertRealTime():
    if(request.method == "POST"):
        stock_name = request.form["stock_name"]
        thres = request.form["thres"]
        ans = stock_alert(stock_name, thres)
        return redirect(url_for("user", usr = ans))
    return render_template("alerts.html")

@app.route("/comparison_stocks.html", methods=["POST", "GET"])
def Comparison_stocks():
    if(request.method == "POST"):
        stock_name = request.form["stock_name"]
        interval = request.form["interval"]
        ans = stock_comparison(stock_name,interval)
        return redirect(url_for("user", usr = ans))
    return render_template("comparison_stocks.html")

@app.route("/technical_charts.html", methods=["POST", "GET"])
def Technical_Charts():
    if(request.method == "POST"):
        stock_name = request.form["stock_name"]
        ans = technical_charts(stock_name)
        return redirect(url_for("user", usr = ans))
    return render_template("technical_charts.html")

# @app.route("/technical_indicators.html", methods=["POST", "GET"])
# def Technical_Indicators():
#     if(request.method == "POST"):
#         stock_name = request.form["stock_name"]
#         interval = request.form["interval"]
#         ans = technical_indicators(stock_name,interval)
#         return redirect(url_for("user", usr = ans))
#     return render_template("technical_indicators.html")
#
# @app.route("/technical_charts.html", methods=["POST", "GET"])
# def Technical_Charts():
#     if(request.method == "POST"):
#         """stock_name = request.form["stock_name"]
#         interval = request.form["interval"]
#         ans = technical_indicators(stock_name,interval)"""
#         ans = technical_charts()
#         return redirect(url_for("user", usr = ans))
#     return render_template("technical_charts.html")

@app.route("/index.html", methods=["POST", "GET"])
def Home():
    return render_template("index.html")

@app.route("/login.html", methods=["POST", "GET"])
def Login():
    return render_template("login.html")

@app.route("/signup.html", methods=["POST", "GET"])
def SignUp():
    return render_template("signup.html")

@app.route("/services.html", methods=["POST", "GET"])
def Services():
    return render_template("services.html")

@app.route("/blogs.html", methods=["POST", "GET"])
def Blogs():
    return render_template("blogs.html")


@app.route("/about.html", methods=["POST", "GET"])
def AboutUs():
    return render_template("about.html")

@app.route("/contact.html", methods=["POST", "GET"])
def ContactUs():
    return render_template("contact.html")

@app.route("/one-page.html", methods=["POST", "GET"])
def Page():
    return render_template("one-page.html")

@app.route("/fundamental_analysis.html", methods=["POST", "GET"])
def Fundamental_Analysis():
    return render_template("fundamental_analysis.html")

@app.route("/technical_analysis.html", methods=["POST", "GET"])
def Technical_Analysis():
    return render_template("technical_analysis.html")

@app.route("/sentiment_analysis.html", methods=["POST", "GET"])
def Sentiment_Analysis():
    return render_template("sentiment_analysis.html")

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"


if(__name__=="__main__"):
    app.run(debug=True)


