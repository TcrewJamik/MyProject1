import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import numpy as np
from sklearn.model_selection import cross_val_score

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

# 1. Обработка выбросов (ограничение 99-м процентилем)
for col in ['Temperature', 'Humidity', 'Precipitation (%)', 'Atmospheric Pressure']:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)

# 2. Подготовка целевой переменной
y = df['Weather Type']
X = df.drop('Weather Type', axis=1)

# 3. Кодирование целевой переменной
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 4. Разделение данных (только для обучения модели)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Подготовка числовых и категориальных признаков
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
categorical_features = ['Cloud Cover', 'Season', 'Location']

# --- Создание входного датафрейма ---
data = {
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
input_df = pd.DataFrame(data, index=[0])

# --- Предобработка входных данных и X ---

# One-Hot Encoding для категориальных признаков
ohe = OneHotEncoder(handle_unknown='ignore')
encoded_categorical_features = ohe.fit_transform(df[categorical_features]).toarray()
X_encoded = pd.DataFrame(encoded_categorical_features, columns=ohe.get_feature_names_out(categorical_features))
X = pd.concat([df[numerical_features].reset_index(drop=True), X_encoded], axis=1)

encoded_input_categorical = ohe.transform(input_df[categorical_features]).toarray()
input_encoded = pd.DataFrame(encoded_input_categorical, columns=ohe.get_feature_names_out(categorical_features))
input_row = pd.concat([input_df[numerical_features].reset_index(drop=True), input_encoded], axis=1)

# Масштабирование числовых признаков
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
input_row[numerical_features] = scaler.transform(input_row[numerical_features]) # применяем к входящим данным

with st.expander('Предобработка данных'):
    st.write('**Входные данные (новая строка)**')
    st.dataframe(input_row)
    st.write('**Объединенные данные (входные данные + исходные данные)**')
    st.dataframe(pd.concat([input_row, pd.DataFrame(X, columns=input_row.columns)], axis=0))
    st.write('**Целевая переменная (y)**')
    st.write(y)

# --- Обучение модели ---
st.subheader('Обучение модели')

# Определение моделей с фиксированными гиперпараметрами
rf_model = RandomForestClassifier(
    bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=50, random_state=42
)
knn_model = KNeighborsClassifier(metric='manhattan', n_neighbors=5, weights='distance')
catboost_model = CatBoostClassifier(depth=6, iterations=100, l2_leaf_reg=3, learning_rate=0.2, random_seed=42, logging_level='Silent')

# Обучение моделей
rf_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)

# Прогнозирование
prediction_rf = rf_model.predict(input_row)
prediction_knn = knn_model.predict(input_row)
prediction_catboost = catboost_model.predict(input_row)

# Вывод результатов
st.subheader('Результаты прогнозирования')

# Прогнозируемый тип погоды для каждой модели
weather_types = label_encoder.classes_

predicted_weather_rf = weather_types[prediction_rf[0]]
predicted_weather_knn = weather_types[prediction_knn[0]]
predicted_weather_catboost = weather_types[prediction_catboost[0]]

st.write(f"**Random Forest Прогноз**: {predicted_weather_rf}")
st.write(f"**K-Nearest Neighbors Прогноз**: {predicted_weather_knn}")
st.write(f"**CatBoost Прогноз**: {predicted_weather_catboost}")

# Вероятности для каждого класса погоды
prediction_proba_rf = rf_model.predict_proba(input_row)
prediction_proba_knn = knn_model.predict_proba(input_row)
prediction_proba_catboost = catboost_model.predict_proba(input_row)

df_prediction_proba_rf = pd.DataFrame(prediction_proba_rf, columns=weather_types)
df_prediction_proba_knn = pd.DataFrame(prediction_proba_knn, columns=weather_types)
df_prediction_proba_catboost = pd.DataFrame(prediction_proba_catboost, columns=weather_types)

st.write('**Вероятности для модели Random Forest:**')
st.dataframe(df_prediction_proba_rf)

st.write('**Вероятности для модели K-Nearest Neighbors:**')
st.dataframe(df_prediction_proba_knn)

st.write('**Вероятности для модели CatBoost:**')
st.dataframe(df_prediction_proba_catboost)
