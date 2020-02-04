import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.drop('NA_Sales', axis = 1)
base = base.drop('EU_Sales', axis = 1)
base = base.drop('JP_Sales', axis = 1)

base = base.dropna(axis = 0)

base = base.loc[base['Global_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name
base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
valor_vendas = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8])
previsores = onehotencoder.fit_transform(previsores).toarray()

entrada = Input(shape=(99,))
at = Activation(activation = 'sigmoid')
camada_oculta1 = Dense(units = 50, activation=at)(entrada)
camada_oculta2 = Dense(units = 50, activation=at)(camada_oculta1)
saida = Dense(units = 1, activation='linear')(camada_oculta2)

regressor = Model(inputs = entrada, outputs=[saida])
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(previsores, valor_vendas, epochs = 5000, batch_size=100)
previs = regressor.predict(previsores)


