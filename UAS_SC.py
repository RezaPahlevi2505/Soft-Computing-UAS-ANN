# Data preprocessing

# Impor libraries yang dibutuhkan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#----------------Data Preprocessing--------------------

# Import dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#data dari kolom ke-3 sampai ke-12
X = dataset.iloc[:, 3:13].values

#jumlah data (1 or 0)
y = dataset.iloc[:, 13].values 

# mengubah data kategorikal ke data numerik
# karena ANN hanya bisa bekerja pada data numerik
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Setelah ini, kita akan lihat kolom countries berubah menjadi angka
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Menghilangkan variabel dummy
X = X[:, 1:]

# Membagi dataset menjadi data latih dan data uji coba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#  ------------------Pembuatan ANN-------------------------
# Import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#Mendefinisikan ANN
classifier = Sequential()

# menambah input layer and hidden layer no.1
# 6 output nodes, Relu activation function and 11 input nodes
# Output node ditentukan dari jumlah input nodes+1/2
# pastikan bobotnya diberikan nomor acak dengan nomor kecil yang mendekati nol

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Menambahkan hidden layer kedua untuk mencapai deep neural network
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))


# Menambahkan Output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# Mengeksekusi NN
# binary_crossentropy loss function digunakan jika output biner tersebut ada
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 


classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Menyesuaikan classifier ke data latih
# Kita akan membuat classifier di sini

# Memprediksi hasil data uji coba
y_pred = classifier.predict(X_test)

# Create a treshold to predict a true or false for leaving the
# the bank.
y_pred = (y_pred > 0.5)

# Membuat Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

##  1545 + 136 prediksi benar and 230 + 50 prediksi salah

# akurasi komputasi 1545 + 136 / 2000 prediksi == 0.8405 % akurat.
