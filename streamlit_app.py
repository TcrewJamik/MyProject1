import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

# --- Загрузка данных ---
data = pd.read_csv("weather.csv")  # Замените "weather.csv"

# --- Предобработка данных ---
# 1. Обработка выбросов (ограничение 99-м процентилем)
for col in ['Temperature', 'Humidity', 'Precipitation (%)', 'Atmospheric Pressure']:
    cap = data[col].quantile(0.99)
    data[col] = data[col].clip(upper=cap)

# 2. Подготовка целевой переменной
y = data['Weather Type']
X = data.drop('Weather Type', axis=1)

# 3. Кодирование целевой переменной
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# --- Streamlit UI ---
st.title("Weather Prediction Model Evaluation")

# --- Настройка признаков ---
st.sidebar.header("Select Features")

# Select numerical features
numerical_features = st.sidebar.multiselect("Select numerical features", options=X.select_dtypes(include=['float64', 'int64']).columns.tolist(), default=['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)'])

# Select categorical features
categorical_features = st.sidebar.multiselect("Select categorical features", options=X.select_dtypes(include=['object']).columns.tolist(), default=['Cloud Cover', 'Season', 'Location'])

# --- Разделение данных ---
X_train, X_test, y_train, y_test = train_test_split(X[numerical_features + categorical_features], y, test_size=0.2, random_state=42)

# --- Кодирование категориальных признаков ---
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_cat_encoded = ohe.fit_transform(X_train[categorical_features]).toarray()
X_test_cat_encoded = ohe.transform(X_test[categorical_features]).toarray()

# --- Объединение числовых и закодированных категориальных признаков ---
X_train = pd.concat([X_train.reset_index(drop=True)[numerical_features],
                     pd.DataFrame(X_train_cat_encoded, columns=ohe.get_feature_names_out(categorical_features))], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True)[numerical_features],
                    pd.DataFrame(X_test_cat_encoded, columns=ohe.get_feature_names_out(categorical_features))], axis=1)

# --- Масштабирование признаков ---
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# --- Оценка моделей ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

# --- Логистическая регрессия ---
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_accuracy, lr_report, lr_conf_matrix = evaluate_model(lr_model, X_test, y_test)
st.subheader("Логистическая регрессия")
st.write(f"Accuracy: {lr_accuracy:.4f}")
st.text(lr_report)
st.write(f"Confusion Matrix:\n{lr_conf_matrix}")

# --- K-ближайших соседей (KNN) ---
knn_model = KNeighborsClassifier()
param_grid = {'n_neighbors': st.sidebar.slider("Select number of neighbors for KNN", min_value=3, max_value=11, step=2, value=5)}
grid_search = GridSearchCV(knn_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
knn_accuracy, knn_report, knn_conf_matrix = evaluate_model(best_knn, X_test, y_test)
st.subheader("K-ближайших соседей")
st.write(f"Accuracy: {knn_accuracy:.4f}")
st.text(knn_report)
st.write(f"Confusion Matrix:\n{knn_conf_matrix}")

# --- Случайный лес (Random Forest) ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy, rf_report, rf_conf_matrix = evaluate_model(rf_model, X_test, y_test)
st.subheader("Случайный лес")
st.write(f"Accuracy: {rf_accuracy:.4f}")
st.text(rf_report)
st.write(f"Confusion Matrix:\n{rf_conf_matrix}")

# --- CatBoost ---
catboost_model = CatBoostClassifier(iterations=50, random_seed=42, logging_level='Silent')
catboost_model.fit(X_train, y_train)
catboost_accuracy, catboost_report, catboost_conf_matrix = evaluate_model(catboost_model, X_test, y_test)
st.subheader("CatBoost")
st.write(f"Accuracy: {catboost_accuracy:.4f}")
st.text(catboost_report)
st.write(f"Confusion Matrix:\n{catboost_conf_matrix}")

# --- Градиентный бустинг ---
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_accuracy, gb_report, gb_conf_matrix = evaluate_model(gb_model, X_test, y_test)
st.subheader("Градиентный бустинг")
st.write(f"Accuracy: {gb_accuracy:.4f}")
st.text(gb_report)
st.write(f"Confusion Matrix:\n{gb_conf_matrix}")

# --- Сводка ---
st.subheader("Сводка по точности моделей")
st.write(f"Логистическая регрессия: {lr_accuracy:.4f}")
st.write(f"K-ближайших соседей: {knn_accuracy:.4f}")
st.write(f"Случайный лес: {rf_accuracy:.4f}")
st.write(f"CatBoost: {catboost_accuracy:.4f}")
st.write(f"Градиентный бустинг: {gb_accuracy:.4f}")
