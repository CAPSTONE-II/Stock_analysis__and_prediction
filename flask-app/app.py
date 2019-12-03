from flask import Flask, render_template, send_from_directory, request
# import numpy as np
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
import json
# from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

# @app.route('/google')
# def predict_google():
# 	# name = request.args.get("name")
# 	dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# 	training_set = dataset_train.iloc[:, 1:2].values
# 	sc = MinMaxScaler(feature_range = (0, 1))
# 	training_set_scaled = sc.fit_transform(training_set)

# 	model = tf.keras.models.load_model('saved_model.h5')

# 	dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
# 	#only the open values of the test data set
# 	real_stock_price = dataset_test.iloc[:, 1:2].values
# 	# print(real_stock_price)

# 	# Getting the predicted stock price of 2017
# 	dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# 	inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values #1199 to end
# 	inputs = inputs.reshape(-1,1)
# 	inputs = sc.transform(inputs)
# 	X_test = []

# 	for i in range(60, 80):
# 	    X_test.append(inputs[i-60:i, 0])
# 	    # print(inputs[i-60:i, 0])
# 	X_test = np.array(X_test)
# 	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# 	predicted_stock_price = model.predict(X_test)
# 	predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# 	result = json.dumps(predicted_stock_price.tolist())

# 	plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
# 	plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
# 	plt.title('Google Stock Price Prediction')
# 	plt.xlabel('Time')
# 	plt.ylabel('Google Stock Price')
# 	plt.legend()
	# plt.savefig("static\images\google.jpg")
	# return render_template('google.html', google_image = "static\images\google.jpg")

@app.route('/get_ticker')
def get_ticker():
   return render_template('ticker_form.html')


@app.route('/time_series',methods = ['POST', 'GET'])
def time_series():
   if request.method == 'POST':
	   ticker = request.form['ticker']
	   ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
	   data, meta_data = ts.get_intraday(symbol=ticker,interval='1min', outputsize='full')
	   data['4. close'].plot()
	   plt.title('Closing price Intraday Times Series')
	   plt.savefig("static\\images\\time.jpg")
	   return render_template('time-series.html')
		


# @app.route('/time_series/<ticker>')
# def time_series(ticker):
# 	ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
# 	data, meta_data = ts.get_intraday(symbol=ticker,interval='1min', outputsize='full')
# 	data['4. close'].plot()
# 	plt.title('Closing price Intraday Times Series')
# 	plt.savefig("static\\images\\time.jpg")
# 	return render_template('time-series.html')

    



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')