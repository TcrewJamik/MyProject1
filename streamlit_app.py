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


st.title('🌦️ Предсказание погоды')

st.write('Здесь мы обучим модель машинного обучения для предсказания типа погоды.')

# --- Загрузка данных ---
file_path = https://raw.githubusercontent.com/TcrewJamik/MyProject1/refs/heads/master/weather_classification_data.csv
df = pd.read_file(file_path)

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
input_row[numerical_features] = scaler.transform(input_row[numerical_features]) #применяем к входящим данным

with st.expander('Предобработка данных'):
    st.write('**Входные данные (новая строка)**')
    st.dataframe(input_row)
    st.write('**Объединенные данные (входные данные + исходные данные)**')
    st.dataframe(pd.concat([input_row, pd.DataFrame(X, columns=input_row.columns)], axis=0))
    st.write('**Целевая переменная (y)**')
    st.write(y)

# --- Обучение модели ---
st.subheader('Обучение модели')

# Определение параметров для Grid Search
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}
param_grid_knn = {
    'n_neighbors': [3, 5, 7]
}
param_grid_catboost = {
    'iterations': [50, 100],
    'learning_rate': [0.01, 0.1],
    'depth': [4, 6]
}

# Base models
base_rf = RandomForestClassifier(random_state=42)
base_knn = KNeighborsClassifier()
base_catboost = CatBoostClassifier(random_seed=42, logging_level='Silent')

# Perform grid search for each model
grid_search_rf = GridSearchCV(base_rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X, y)

grid_search_knn = GridSearchCV(base_knn, param_grid_knn, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(X, y)

grid_search_catboost = GridSearchCV(base_catboost, param_grid_catboost, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_catboost.fit(X, y)

# Find the best model based on accuracy
best_rf = grid_search_rf.best_estimator_
best_knn = grid_search_knn.best_estimator_
best_catboost = grid_search_catboost.best_estimator_

models = {
    'Random Forest': best_rf,
    'K-Nearest Neighbors': best_knn,
    'CatBoost': best_catboost
}

best_model_name = max(models, key=lambda k: cross_val_score(models[k], X, y, cv=3).mean())
best_model = models[best_model_name]
best_params = {}
if best_model_name == 'Random Forest':
    best_params = grid_search_rf.best_params_
elif best_model_name == 'K-Nearest Neighbors':
    best_params = grid_search_knn.best_params_
elif best_model_name == 'CatBoost':
    best_params = grid_search_catboost.best_params_

st.write(f"**Лучшая модель:** {best_model_name}")
st.write(f"**Лучшие параметры:** {best_params}")

# --- Прогнозирование ---
prediction = best_model.predict(input_row)
prediction_proba = best_model.predict_proba(input_row)

# --- Вывод результатов ---
st.subheader('Результаты прогнозирования')

# Assuming your label encoder has a classes_ attribute
weather_types = label_encoder.classes_

# Create a DataFrame for the prediction probabilities
df_prediction_proba = pd.DataFrame(prediction_proba, columns=weather_types)

st.write('**Вероятности для каждого класса погоды:**')
st.dataframe(
    df_prediction_proba,
    column_config={
        weather_type: st.column_config.ProgressColumn(
            weather_type,
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ) for weather_type in weather_types
    },
    hide_index=True
)

# Map prediction index back to weather type name
predicted_weather_type = weather_types[prediction[0]]

st.success(f"Прогнозируемый тип погоды: **{predicted_weather_type}**")
