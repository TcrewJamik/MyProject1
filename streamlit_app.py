import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import numpy as np

st.title('🌦️ Предсказание погоды')

st.write('Здесь мы обучим модель машинного обучения для предсказания типа погоды.')

# --- Загрузка данных ---
file_path = r"https://raw.githubusercontent.com/TcrewJamik/MyProject1/refs/heads/master/weather_classification_data.csv"
df = pd.read_csv(file_path)

with st.expander('Данные'):
    st.write("X")
    X_raw = df.drop('Weather Type', axis=1)
    st.dataframe(X_raw)

    st.write("y")
    y_raw = df['Weather Type']
    st.dataframe(y_raw)

# --- Боковая панель ---
with st.sidebar:
    st.header("Введите признаки погоды: ")
    temperature = st.slider('Temperature', -25.0, 45.0, 15.0)
    humidity = st.slider('Humidity', 20, 100, 60)
    wind_speed = st.slider('Wind Speed', 0.0, 25.0, 5.0)
    precipitation = st.slider('Precipitation (%)', 0.0, 100.0, 0.0)
    cloud_cover = st.selectbox('Cloud Cover', ('partly cloudy', 'sunny', 'cloudy'))
    atmospheric_pressure = st.slider('Atmospheric Pressure', 980.0, 1030.0, 1013.0)
    uv_index = st.slider('UV Index', 0, 10, 2)
    season = st.selectbox('Season', ('Spring', 'Summer', 'Autumn', 'Winter'))
    visibility = st.slider('Visibility (km)', 0.0, 20.0, 10.0)
    location = st.selectbox('Location', ('London', 'New York', 'Tokyo', 'Sydney', 'Dubai'))

# --- Визуализация данных ---
st.subheader('Визуализация данных')

fig = px.scatter(
    df,
    x='Temperature',
    y='Humidity',
    color='Season',
    title='Температура vs. Влажность по сезонам'
)
st.plotly_chart(fig)

fig2 = px.histogram(
    df,
    x='Wind Speed',
    nbins=30,
    title='Распределение скорости ветра'
)
st.plotly_chart(fig2)

# --- Предобработка данных ---
# 1. Обработка выбросов
for col in ['Temperature', 'Humidity', 'Precipitation (%)', 'Atmospheric Pressure']:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)

# 2. Подготовка целевой переменной
y = df['Weather Type']
X = df.drop('Weather Type', axis=1)

# 3. Кодирование целевой переменной
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 4. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Подготовка числовых и категориальных признаков
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
categorical_features = ['Cloud Cover', 'Season', 'Location']

# --- Создание входного датафрейма ---
input_data = {
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind Speed': wind_speed,
    'Precipitation (%)': precipitation,
    'Cloud Cover': cloud_cover,
    'Atmospheric Pressure': atmospheric_pressure,
    'UV Index': uv_index,
    'Season': season,
    'Visibility (km)': visibility,
    'Location': location
}
input_df = pd.DataFrame(input_data, index=[0])

# --- Предобработка входных данных и X ---

# One-Hot Encoding для категориальных признаков
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_cat_encoded = ohe.fit_transform(X_train[categorical_features]).toarray()
X_test_cat_encoded = ohe.transform(X_test[categorical_features]).toarray()

# Объединение числовых и закодированных категориальных признаков для X_train и X_test
X_train = pd.concat(
    [X_train.reset_index(drop=True)[numerical_features],
     pd.DataFrame(X_train_cat_encoded, columns=ohe.get_feature_names_out(categorical_features))],
    axis=1
)
X_test = pd.concat(
    [X_test.reset_index(drop=True)[numerical_features],
     pd.DataFrame(X_test_cat_encoded, columns=ohe.get_feature_names_out(categorical_features))],
    axis=1
)
#аналогично для входящих данных
encoded_input_categorical = ohe.transform(input_df[categorical_features]).toarray()
input_encoded = pd.DataFrame(encoded_input_categorical, columns=ohe.get_feature_names_out(categorical_features))
input_row = pd.concat([input_df[numerical_features].reset_index(drop=True), input_encoded], axis=1)

# Масштабирование числовых признаков
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
input_row[numerical_features] = scaler.transform(input_row[numerical_features]) #применяем к входящим данным

# --- Обучение модели ---
st.subheader('Обучение модели')

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- Прогнозирование ---
prediction_rf = rf_model.predict(input_row)

# --- Вывод результатов ---
st.subheader('Результаты прогнозирования')
predicted_weather_rf = label_encoder.inverse_transform(prediction_rf)
st.write(f"**Random Forest Прогноз**: {predicted_weather_rf[0]}")
