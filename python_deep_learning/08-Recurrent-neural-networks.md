# Recurrent Neural Networks for Timeseries forecasting

## Recurrent Neural Networks

### Introduction
- RNNs are a type of deep learning model with a built-in feedback mechanism.
- The output of a particular layer can be **re-fed** as the input in order to predict the output.

![image](https://user-images.githubusercontent.com/43855029/132912049-167cf37e-66a0-4b54-8024-183ab7785398.png)

[Understanding LSTMS](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- A detailed look at unrolling the RNN loop:

![image](https://user-images.githubusercontent.com/43855029/132911838-0ce7eb99-fd60-44c7-b554-d176fdb45f8b.png)


### Types of RNNs

![image](https://user-images.githubusercontent.com/43855029/132903689-398ef108-660d-47ba-ae46-b783f203e307.png)

### Applications
- It is specifically designed for sequential problems like **weather forecasta, stock forecasta, image captioning, natural language processing, and speech/voice recognition**

### Some Disadvantages of RNNs:
- Computationally expensive and require a large memory.
- RNNs are sensitive to changes in parameters and both **exploding gradients** or **vanishing gradients**
- In order to resolve the gradient problems of RNNs, we can use a method called Long-Short Term Memory (LSTM).

In this limited workshop, we will only cover LSTM for timeseries forecast problems.

## Long-Short Term Memory model - LSTM
### Introduction
- LSTMs are a special kind of RNN — They are capable of learning long-term dependencies by remembering information for long periods..
- They were introduced by Hochreiter & Schmidhuber (1997) and were refined and popularized by many people
- LSTMs are explicitly designed to avoid the long-term dependency problem.

### Comparison between traditional RNNs and LSTMs

![image](https://user-images.githubusercontent.com/43855029/132913273-1b7d4765-a8f2-4f2d-b3b9-6910d5d15807.png)

### Step by step walkthrough of LSTMs:
[Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

# Using LSTMs in temperature forecasting
We will access Keras LSTM to forecast temperatures at a site named Jena (Germany), given information about temperature and other climate variables.
This tutorial follows the [Keras website tutorial](https://keras.io/examples/timeseries/timeseries_weather_forecasting/), but has been rewritten in a simpler way for ease of understanding.

### Climate Data
- Data comes from a single station: Jena, Germany.
- Data consists of 14 climate variables recorded every 10 minutes
- 8 year timeframe: 01/01/2009 - 12/31/2016
- Data description:


![image](https://user-images.githubusercontent.com/43855029/132914704-b2c7ee79-0c99-482a-abfd-cc4575dcfe1b.png)

- Input variables: all 14 climate variables
- Output or target variable: Temperature at a later date

### Objective
- Using data from previous 5 days, forecast the temperature in the next 12 hours

#### Loading library:

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from numpy import array
```

#### Loading Jthe ena climate station data:

```python
df = pd.read_csv("/zfs/citi/workshop_data/python_ml/jena_climate_2009_2016.csv")
```

#### Check for any missing values

```python
#Check missing value
print(df.isnull().sum())
print(df.isna().sum())
print(df.min())
```

There are missing values for wv and max. wv (denoted by -9999). Therefore we need to convert -9999 to nan

```python
df1 = df.copy()
#Convert -9999 to nan
df1[df1==-9999.0]=np.nan
print(df1.isna().sum())
```

Now treat missing value with KNN Imputer


```python
#Treat missing values using KNN Imputer method
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=15, weights="uniform")
df_knnimpute = pd.DataFrame(imputer.fit_transform(df1.iloc[:,1:]))
df_knnimpute.columns=df.columns[1:]
print(df_knnimpute.isna().sum())
```

Now all input data is clean without any missing values. Next, we use LASSO for variable selection:

#### Variable selection with LASSO

Create set of input/output data.
Here, the output variable is "T (degC)". However "Tpot (K)" and "Tdew (degC)" are very similar to the output, resulting in collinearity. Drop them for now in order to check the influence of the variables with the output:

```python
x = df_knnimpute.drop(['T (degC)','Tpot (K)','Tdew (degC)'],1)
y = df_knnimpute.loc[:,"T (degC)"]
```

Apply LASSO to select the most influential input variables with output:

```python
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse

n_lambda = 100
lambdas = np.logspace(-6,0, n_lambda)

MSE = []
coefs = []
for ld in lambdas:
    lassocv = Lasso(alpha=ld)
    model_LS = lassocv.fit(x, y)
    y_predLS_cv = model_LS.predict(x)
    MSE.append(mse(y,y_predLS_cv))
    coefs.append(model_LS.coef_)
```

Plot the MSE with a lambda variation:

```python
plt.scatter(np.log10(lambdas), MSE,color="red")
plt.title("MSE with Regularization Penalty $\\lambda$ variation ")
plt.xlabel("log($\\lambda$)")
plt.ylabel('MSE')
plt.show()
```

![image](https://user-images.githubusercontent.com/43855029/133639140-80d66516-ebfd-4142-bddf-f419c49da276.png)

Plot the corresponding coefficients with varying lambda variations:

```python
coef_df = pd.DataFrame(coefs)
coef_df.columns = x.columns

ax = plt.gca()
for i in range(0,coef_df.columns.size):
    ax.plot(np.log10(lambdas), coef_df.iloc[:,i])

ax.legend(coef_df.columns,bbox_to_anchor = (1.05, 0.6))
plt.xlabel("log($\\lambda$)")
plt.ylabel('Coefficients')
plt.title('LASSO Coefficients')
plt.axis('tight')
plt.show()
```

![image](https://user-images.githubusercontent.com/43855029/133639475-34df38f2-8435-49dd-8fcc-a1997c5e1b3b.png)

From the MSE vs lambda plot, log lambda value = -1.2 is the elbow curve for the variable selection.
The corresponding coefficient with log(lambda)=-1.2 is:

```python
ind = np.abs((np.log10(lambdas)+1.2)).argmin()
coef_df.iloc[ind]
```

```
p (mbar)           0.249129
rh (%)            -0.011646
VPmax (mbar)       0.092128
VPact (mbar)       0.000000
VPdef (mbar)       0.011752
sh (g/kg)          0.000000
H2OC (mmol/mol)    0.000000
rho (g/m**3)      -0.199438
wv (m/s)          -0.000000
max. wv (m/s)     -0.000000
wd (deg)           0.000000
Name: 79, dtype: float64
```

Here we see that the variables 'p (mbar)', 'rh (%)', 'VPmax (mbar)', 'rho (g/m**3)' also have good influence on the output.
Therefore, we select all these variables into our input data together with T (degC):

```python
selected_col = [0,1,4,5,10]
dfnew = df_knnimpute.iloc[:,selected_col]
dfnew.head()
```
![image](https://user-images.githubusercontent.com/43855029/133648740-4a4d8987-30d6-4ea1-b5a5-d814b4c81c1e.png)

#### Data partitioning

- Data was collected at 10 minute intervals or 6 times an hour. Resample the input data to hourly with the **sampling_rate** argument: **step=6**
- Use the historical data of the past 5 days: 5 x 24 x 6 = **720 data points**
- To forecast temperature in the next 12 hours: 12 x 6 = **72 data points**
- Data partition to **70% training** and **30% testing** in order of time
- For neural networks, the following parameters are pre-selected:
   - **Learning rate** = 0.001
   - **Batch size** = 256
   - **Epoch** = 10

```python
split_fraction = 0.7
train_split = int(split_fraction * int(df.shape[0]))

step = 6
past = 720
future = 72

learning_rate = 0.0001
batch_size = 256
epochs = 10
```

Since the input data has a different range, you will need to perform **standardization**

```python
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range=(0,1))
scaled_features = pd.DataFrame(scale.fit_transform(dfnew))
scaled_features.columns = dfnew.columns
scaled_features.index = dfnew.index

train_data = scaled_features[0:train_split]
test_data =  scaled_features[train_split:]

train_data.head()
```

![image](https://user-images.githubusercontent.com/43855029/133648834-659d6af7-95de-4969-9a03-c6f1f49d4846.png)

#### Selecting input/output for training/testing dataset:

![image](https://user-images.githubusercontent.com/43855029/133829483-8190525b-d278-497c-af5c-73f5a372855d.png)

##### Training

```python
start_ytrain = past + future
end_ytrain = train_split + start_ytrain

x_train = train_data
y_train = scaled_features[start_ytrain:end_ytrain]["T (degC)"]

sequence_length = int(past/step)
```

##### Testing

```python
start_ytest = end_ytrain
end_ytest = len(test_data) - past - future

x_test = test_data.iloc[:end_ytest,:]
y_test = scaled_features.iloc[start_ytest:]["T (degC)"]
```


For training data sets, the updated Keras (with Tensorflow version 2.3 and above) has built-in functions to prepare for time series modeling using given batch size and given length for historical data.

```python
dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate = step,
    batch_size=batch_size
)
```

#### Using Keras to split training/testing data into different batches:
Here, we use the preprocessing time series feature of Keras to split training/testing data into different batches:

##### Training

```python
dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate = step,
    batch_size=batch_size,
)
```

```python
for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)
```

```
Input shape: (256, 120, 5)
Target shape: (256,)
```

##### Testing

```python
dataset_test = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_test,
    y_test,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size
)
```

```python
for batch in dataset_test.take(1):
    inputs_test, targets_test = batch

print("Input shape:", inputs_test.numpy().shape)
print("Target shape:", targets_test.numpy().shape)
```

```
Input shape: (256, 120, 5)
Target shape: (256,)
```

#### Build a deep learning model with an LSTM framework:

```python
inputs = tf.keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = tf.keras.layers.LSTM(32, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(1)(lstm_out)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=['accuracy'])
model.summary()
```

```
Model: "functional_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_3 (InputLayer)         [(None, 120, 5)]          0
_________________________________________________________________
lstm_2 (LSTM)                (None, 32)                4864
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33
=================================================================
Total params: 4,897
Trainable params: 4,897
Non-trainable params: 0
```

#### Train the LSTM model and vaidate with testing data set:

```python
history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_test
)
```


#### Visualize the training & testing loss with 10 different epochs

```python
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

visualize_loss(history, "Training and Validation Loss")
```

![image](https://user-images.githubusercontent.com/43855029/133833536-c25e5c6d-8fff-4509-9786-aec2f94fd201.png)

#### Save & Load the LSTM trained model

Save LSTM model:
```python
model.save('LSTM_Jena.keras')
```

Load LSTM model

```python
model = tf.keras.models.load_model('LSTM_Jena.keras')
```


#### Prediction
Modify the given [Timeseries forecasting code](https://keras.io/examples/timeseries/timeseries_weather_forecasting/) to make predictions for 5 sets of values from the validation set:

First, we need to create a rescale function back to original scale for `T (degC)`

```python
#Create transformation function to rescale back to original
scaleT = MinMaxScaler(feature_range=(0,1))
scaleT.fit_transform(pd.DataFrame(dfnew[:]["T (degC)"]))
```

Apply plotting:

```python
def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.ylabel("T (degC)")
    plt.show()
    return


for x, y in dataset_test.take(5):
    show_plot(
        #[x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
        [scaleT.inverse_transform(pd.DataFrame(x[0][:, 1])),
         scaleT.inverse_transform(pd.DataFrame(pd.Series(y[0].numpy()))),
         scaleT.inverse_transform(pd.DataFrame(model.predict(x)[0]))],
        12,
        "Single Step Prediction",
    )
```
![image](https://user-images.githubusercontent.com/43855029/133819908-9426fe72-7a6e-4b13-8738-82c041b3ca7d.png)

