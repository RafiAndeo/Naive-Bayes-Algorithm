"""__________________________________________________________________
Python Program for Naive Bayes Learning                              |
Kelompok 2:                                                          |
DIAN RAMADHINI (1301200254)                                          |
MUHAMMAD RAFI ANDEO PRAJA (1301200278)                               |
_____________________________________________________________________|"""

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read excel data
xls = pd.ExcelFile('traintest.xlsx')
df_train = pd.read_excel(xls, 'train')
df_train_1 = pd.read_excel(xls, 'train y = 1')
df_train_0 = pd.read_excel(xls, 'train y = 0')
df_test = pd.read_excel(xls, 'test')

# drop id
df_train_1 = df_train_1.drop('id', axis=1)
df_train_0 = df_train_0.drop('id', axis=1)
df_test = df_test.drop('id', axis=1)

# new variable for each x with output 0
df_train_0_x1 = df_train_0['x1']
df_train_0_x2 = df_train_0['x2']
df_train_0_x3 = df_train_0['x3']

# new variable for each x with output 1
df_train_1_x1 = df_train_1['x1']
df_train_1_x2 = df_train_1['x2']
df_train_1_x3 = df_train_1['x3']

# new variable for each x in test section
df_test_x1 = df_test['x1']
df_test_x2 = df_test['x2']
df_test_x3 = df_test['x3']

# function average for output 0
def average_0(arr):
    i = 0
    total = 0
    while i < 78:
        total = total + arr[i]
        i += 1
    rata = total / i
    return rata

# function average for output 1
def average_1(arr):
    i = 0
    total = 0
    while i < 218:
        total = total + arr[i]
        i += 1
    rata = total / i
    return rata

# assign variable with average function for output 0
rata_x1_0 = average_0(df_train_0_x1)
rata_x2_0 = average_0(df_train_0_x2)
rata_x3_0 = average_0(df_train_0_x3)

# assign variable with average function for output 1
rata_x1_1 = average_1(df_train_1_x1)
rata_x2_1 = average_1(df_train_1_x2)
rata_x3_1 = average_1(df_train_1_x3)

# new variable for stdev output 0
stdev_x1_0 = np.std(df_train_0_x1)
stdev_x2_0 = np.std(df_train_0_x2)
stdev_x3_0 = np.std(df_train_0_x3)

# new variable for stdev output 1
stdev_x1_1 = np.std(df_train_1_x1)
stdev_x2_1 = np.std(df_train_1_x2)
stdev_x3_1 = np.std(df_train_1_x3)

# function normal distribution or gauss probability
def normal(x, average, stdev):
    prob = 1 / np.sqrt(2 * np.pi * stdev**2)
    return prob * np.exp(-0.5 / stdev**2 * (x - average)**2)

# function accumulated probability for output 0
def peluang_total_0(peluang_0_x1, peluang_0_x2, peluang_0_x3):
    return 78/296 * peluang_0_x1 * peluang_0_x2 * peluang_0_x3

# function accumulated probability for output 1
def peluang_total_1(peluang_1_x1, peluang_1_x2, peluang_1_x3):
    return 218/296 * peluang_1_x1 * peluang_1_x2 * peluang_1_x3

# new variable with normal function for output 0
peluang_0_x1 = normal(df_test_x1, rata_x1_0, stdev_x1_0)
peluang_0_x2 = normal(df_test_x2, rata_x2_0, stdev_x2_0)
peluang_0_x3 = normal(df_test_x3, rata_x3_0, stdev_x3_0)

# new variable with normal function for output 1
peluang_1_x1 = normal(df_test_x1, rata_x1_1, stdev_x1_1)
peluang_1_x2 = normal(df_test_x2, rata_x2_1, stdev_x2_1)
peluang_1_x3 = normal(df_test_x3, rata_x3_1, stdev_x3_1)

# new variable for accumulated all probability
peluang_0_total = peluang_total_0(peluang_0_x1, peluang_0_x2, peluang_0_x3)
peluang_1_total = peluang_total_1(peluang_1_x1, peluang_1_x2, peluang_1_x3)

# function prediction
def prediksi(peluang_0_total, peluang_1_total):
    hasil = 0
    if peluang_0_total > peluang_1_total:
        hasil = 0
    elif peluang_1_total > peluang_0_total:
        hasil = 1
    return hasil

# training model
plt.title("Training Model")
plt.xlabel("Peluang 0 atau 1")
plt.ylabel("Nilai test x1, x2, x3")
plt.scatter(peluang_0_x1, df_test_x1, color = "red", label = "Peluang Output 0 x1", alpha = 0.3)
plt.scatter(peluang_1_x1, df_test_x1, color = "blue", label = "Peluang Output 1 x1", alpha = 0.3)
plt.scatter(peluang_0_x2, df_test_x2, color = "lime", label = "Peluang Output 0 x2", alpha = 0.3)
plt.scatter(peluang_1_x2, df_test_x2, color = "magenta", label = "Peluang Output 1 x2", alpha = 0.3)
plt.scatter(peluang_0_x3, df_test_x3, color = "black", label = "Peluang Output 0 x3", alpha = 0.3)
plt.scatter(peluang_1_x3, df_test_x3, color = "orange", label = "Peluang Output 1 x3", alpha = 0.3)
plt.legend()
plt.show()

plt.title("Testing Analysis")
plt.xlabel("Peluang Total 0 atau 1")
plt.ylabel("Nilai test x1, x2, x3")
plt.scatter(peluang_0_total, df_test_x1, color = "red", label = "Peluang Output 0 x1", alpha = 0.3)
plt.scatter(peluang_1_total, df_test_x1, color = "blue", label = "Peluang Output 1 x1", alpha = 0.3)
plt.scatter(peluang_0_total, df_test_x2, color = "lime", label = "Peluang Output 0 x2", alpha = 0.3)
plt.scatter(peluang_1_total, df_test_x2, color = "magenta", label = "Peluang Output 1 x2", alpha = 0.3)
plt.scatter(peluang_0_total, df_test_x3, color = "black", label = "Peluang Output 0 x3", alpha = 0.3)
plt.scatter(peluang_1_total, df_test_x3, color = "orange", label = "Peluang Output 1 x3", alpha = 0.3)
plt.legend()
plt.show()

# function accuracy
def akurasi(prediction):
    true_values = np.array([[0, 1, 1, 1, 1, 1, 0, 0, 1, 1]])
    prediction_values = np.array([[prediction]])

    n = true_values.shape[1]
    accuracy = (true_values == prediction_values).sum() / n

    return accuracy

# output to file
i = 0
prediction = [None] * 10
while i < 10:
  prediction[i] = prediksi(peluang_0_total[i], peluang_1_total[i])
  i += 1

print(akurasi(prediction))

Result = pd.DataFrame(prediction, columns=['Testing Output'])
Result.to_excel('Result.xlsx')