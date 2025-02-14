import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
import json
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import pickle
import os
import gzip
from sklearn.metrics import confusion_matrix


test_data = pd.read_csv(
    "files/input/test_data.csv.zip",
    index_col=False,
    compression="zip",
)

train_data = pd.read_csv(
    "files/input/train_data.csv.zip",
    index_col=False,
    compression="zip",
)
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
def clear_data(df):
    df=df.rename(columns={'default payment next month': 'default'})
    df=df.drop(columns=['ID'])
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

    return df
train_data=clear_data(train_data)
test_data=clear_data(test_data)
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

x_train=train_data.drop(columns="default")
y_train=train_data["default"]


x_test=test_data.drop(columns="default")
y_test=test_data["default"]
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.



#Columnas categoricas
categorical_features=["SEX","EDUCATION","MARRIAGE"]

# Combinación de preprocesadores

#preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder=MinMaxScaler()
)

# Creación del Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_regression, k=10)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.


# Definir los hiperparámetros a optimizar
param_grid = {
    "feature_selection__k": range(1, len(x_train.columns) + 1),
    "classifier__C": [0.1, 1, 10],
    "classifier__solver": ["liblinear", "lbfgs"]
}

model=GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True
    )

model.fit(x_train, y_train)
# Paso 5.
# Guarde el modelo como "files/models/model.pkl".


models_dir = 'files/models'
os.makedirs(models_dir, exist_ok=True)

with open("files/models/model.pkl","wb") as file:
    pickle.dump(model,file)

# Compress the pickle file
with gzip.open('files/models/model.pkl.gz', 'wb') as f:
    pickle.dump(model, f)


def metrics(model, X_train, X_test, y_train, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
    }

    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
    }

    output_dir = 'files/output'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'metrics.json')
    with open(output_path, 'w') as f:  # Usar 'w' para comenzar con un archivo limpio
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')

# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}


def matrices(model, X_train, X_test, y_train, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    def format_confusion_matrix(cm, dataset_type):
        return {
            'type': 'cm_matrix',
            'dataset': dataset_type,
            'true_0': {
                'predicted_0': int(cm[0, 0]),
                'predicted_1': int(cm[0, 1])
            },
            'true_1': {
                'predicted_0': int(cm[1, 0]),
                'predicted_1': int(cm[1, 1])
            }
        }

    metrics = [
        format_confusion_matrix(cm_train, 'train'),
        format_confusion_matrix(cm_test, 'test')
    ]

    output_path = 'files/output/metrics.json'
    with open(output_path, 'a') as f:  # Usar 'a' para agregar después de las métricas
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')

# Función principal para ejecutar todo
def main(model, X_train, X_test, y_train, y_test):
    # Crear el directorio de salida si no existe
    
    os.makedirs('files/output', exist_ok=True)

    # Calcular y guardar las métricas
    metrics(model, X_train, X_test, y_train, y_test)

    # Calcular y guardar las matrices de confusión
    matrices(model, X_train, X_test, y_train, y_test)

# Ejemplo de uso:
main(model, x_train, x_test, y_train, y_test)


metrics=[]
with open("files/output/metrics.json", "r", encoding="utf-8") as file:
    for line in file:
        metrics.append(json.loads(line))
metrics