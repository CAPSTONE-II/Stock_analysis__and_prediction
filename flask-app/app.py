from flask import Flask
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import json
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)

@app.route('/')
def predict():
	dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
	training_set = dataset_train.iloc[:, 1:2].values
	sc = MinMaxScaler(feature_range = (0, 1))
	training_set_scaled = sc.fit_transform(training_set)

	model = tf.keras.models.load_model('saved_model.h5')

	dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
	#only the open values of the test data set
	real_stock_price = dataset_test.iloc[:, 1:2].values
	# print(real_stock_price)

	# Getting the predicted stock price of 2017
	dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
	print(dataset_total)
	inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values #1199 to end
	inputs = inputs.reshape(-1,1)
	inputs = sc.transform(inputs)
	X_test = []

	for i in range(60, 80):
	    X_test.append(inputs[i-60:i, 0])
	    print(inputs[i-60:i, 0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


	predicted_stock_price = model.predict(X_test)
	predicted_stock_price = sc.inverse_transform(predicted_stock_price)
	result = json.dumps(predicted_stock_price.tolist())
	return result

    



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')