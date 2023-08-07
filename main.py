import pandas as pd
import numpy as np
import ast
from datetime import datetime
from fastapi import FastAPI

app = FastAPI()

# Definición de los endpoints

@app.get("/genero/")
def genero(anio: str):
    df_gen_anio = df_generos[df_generos.anio == anio]
    lista_generos = []
    for _ in range(5):
        mayor = df_gen_anio.genres.describe().top
        lista_generos.append(mayor)
        df_gen_anio = df_gen_anio[df_gen_anio['genres'] != mayor]

    return {anio: lista_generos}


@app.get("/juegos/")
def juegos(anio: str):
    df_juegos = df[df['anio'] == anio]
    return {anio: list(df_juegos.title)}


@app.get("/specs/")
def specs(anio: str):
    df_specs_anio = df_specs[df_specs.anio == anio]
    lista_specs = []
    for _ in range(5):
        mayor = df_specs_anio.specs.describe().top
        lista_specs.append(mayor)
        df_specs_anio = df_specs_anio[df_specs_anio['specs'] != mayor]

    return {anio: lista_specs}


@app.get("/earlyaccess/")
def earlyaccess(anio: str):
    df_early = df[df['anio'] == anio]
    return {anio: int(df_early[df_early.early_access == True].title.count())}


@app.get("/sentiment/")
def sentiment(anio: str):
    df_sent = df[df['anio'] == anio]
    dicc = {}
    mayor = ''
    cantidad = 0
    for i in range(len(df_sent.sentiment.unique())):
        if df_sent.sentiment.describe().top == 'No data':
            df_sent = df_sent[df_sent['sentiment'] != 'No data']
        else:
            mayor = df_sent.sentiment.describe().top
            cantidad = int(df_sent.sentiment.describe().freq)
            dicc[mayor] = cantidad
            df_sent = df_sent[df_sent['sentiment'] != mayor]

    return dicc


@app.get("/metascore/")
def metascore(anio: str):
    df_meta = df[df['anio'] == anio]
    dicc_mayores = {}

    df_meta_mayor = df_meta.sort_values(by="metascore", ascending=False)
    for _, fila in df_meta_mayor.iterrows():
        if len(dicc_mayores) == 5 or df_meta_mayor.metascore.count() == len(dicc_mayores):
            break
        else:
            dicc_mayores[fila['title']] = fila['metascore']
    if len(dicc_mayores) == 0:
        return "No hay valoraciones registradas este año"

    return dicc_mayores

@app.get("/prediccion/")
def prediccion(publisher, tags, sentiment, anio):
    data_prediccion = pd.DataFrame({
    'publisher': [publisher],
    'tags': [tags],
    'sentiment': [sentiment],
    'anio': [anio]
    })
    
    # Filtro el DataFrame en base a las condiciones de publisher, tags, sentiment y anio
    filtro = (df_ml['publisher'] == publisher) & (df_ml['tags'] == tags) &  (df_ml['sentiment'] == sentiment) &  (df_ml['anio'] == anio)
    
    #Creo un dataframe filtrado para buscar el precio original
    df_filtrado = df_ml.loc[filtro]
    
    # Si el dataframe esta vacio, por ejemplo por poner un anio erroneo, eliminamos el anio y seguimos la busqueda.
    if int(df_filtrado.anio.count()) == 0:
        filtro = (df_ml['publisher'] == publisher) & (df_ml['tags'] == tags) &  (df_ml['sentiment'] == sentiment)
        df_filtrado = df_ml.loc[filtro]
        
    #Hago una segunda comprobacion y si aun asi no hay datos por otro error en los parametros retorno que no se pueden encontrar datos.    
    if int(df_filtrado.anio.count()) == 0:
        return 'Datos no encontrados'
        
    #Busco los valores del precio original para los filtros aplicados y si hay mas de uno calculamos el promedio.
    precio_original = df_filtrado['price'].values
    precio_original = [precio_original.mean()]
    
    precio_predict = linear_model.predict(data_prediccion)
    
    rmse = mean_squared_error(precio_original, precio_predict, squared=False)

    return f'El precio estimado es: {round(precio_predict[0], 2)}, el RMSE es: {round(rmse,2)} y el RMSE del modelo es: {rmse_modelo}'

# Creo esta funcion para convertir la columna release_date en la columna anio siendo solo el numero del anio en formato string.
def convertir_a_anio(fecha_str):
    if pd.notnull(fecha_str):
        try:
            fecha_convertida = datetime.strptime(fecha_str, '%Y-%m-%d')
            anio = str(fecha_convertida.year)
            return anio
        except ValueError:
            return pd.NaT 
    else:
        return pd.NaT 


#Convierto el archivo json en una lista de python por lineas.
lineas_js = []
with open('./src/steam_games.json') as f:
    for line in f.readlines():
        lineas_js.append(ast.literal_eval(line))

df = pd.DataFrame(lineas_js)

# Elimino los registros que contienen precio nulo porque si el objetivo es predecir el precio considero que estos registros no me aportan al entrenamiento de mi modelo.
df = df.dropna(subset=['price'])

# Agrego los registros de developer para completar la columna publisher.
df['publisher'].fillna(df['developer'], inplace=True)

# Relleno los nulos en 'title' con los valores de 'app_name y elimino app_name para evitar redundancias.

df['title'] = df.apply(lambda row: row['app_name'] if pd.isnull(row['title']) else row['title'], axis=1)
df = df.drop(columns='app_name')

#Veo los nulos y borro el de indice 74 porque tiene practicamente toda la informacion NaN
df[df['title'].isnull()]
df = df.drop(74)

#Convierto los registros que indican que un juego es gratis a precio 0.00. Tambien los precios que estan en formato string les aplico el precio correcto.
df['price'] = df['price'].replace(['Free To Play', 'Free to Play', 'Free', 'Free Demo', 'Play for Free!', 'Install Now', 'Play WARMACHINE: Tactics Demo', 'Free Mod', 'Install Theme', 'Third-party', 'Play Now', 'Free HITMAN™ Holiday Pack', 'Play the Demo', 'Free to Try', 'Free Movie', 'Free to Use'], 0.00)
df['price'] = df['price'].replace(['Starting at $499.00'], 499.00)
df['price'] = df['price'].replace(['Starting at $449.00'], 449.00)

#Aplico la funcion de conversion a anio
df['anio'] = df.release_date.apply(convertir_a_anio)


#Expando el dataframe original para ver los distintos generos y creo un dataframe nuevo con las columnas Genero y Anio
df_generos = df.explode('genres')
columnas_a_mantener = ['genres', 'anio']
df_generos = df_generos.drop(columns=[col for col in df_generos.columns if col not in columnas_a_mantener])

#Expando el dataframe original para ver los distintos specs y creo un dataframe nuevo con las columnas a Specs y Anio
df_specs = df.explode('specs')
columnas_a_mantener = ['specs', 'anio']
df_specs = df_specs.drop(columns=[col for col in df_specs.columns if col not in columnas_a_mantener])


#Lleno los datos NaN con un string No data en la columna sentiment para poder trabajar esa columna completa
df.sentiment.fillna("No data", inplace=True)

# Convierto los valores de string 'NA' en valores NaN con el parametro errors=coerce. Esto lo hago para poder tener solo numeros y NaN en la columna.
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')

#Elimino las columnas url, reviews_url, release_date, id y developer porque no van a ser utilizadas.
df.drop(columns=['url', 'reviews_url', 'release_date', 'id', 'developer'], inplace=True)

# ---------------------- FIN FUNC ------------------------------------- #
# ---------------------- INICIO ETL PARA ML ------------------------- #

#Creo un nuevo dataset copiando el original para no perder los datos originales que va a ser utilizado para el proceso de ML.

df_ml = df

# Creo una mascara para reemplazar los registros que contienen 'user' de la columna sentiment con No data. Porque considero que no hay datos suficientes.

mask = df_ml['sentiment'].str.contains('user', case=False)
df_ml.loc[mask, 'sentiment'] = 'No data'

# Lleno los Nan con 0 de la columna discount_price
df_ml['discount_price'].fillna(0, inplace=True)

#Elimino los Nan que quedan en el dataset
df_ml.dropna(subset=['genres'], inplace=True)
df_ml.dropna(subset=['tags'], inplace=True)
df_ml.dropna(subset=['specs'], inplace=True)
df_ml.dropna(subset=['anio'], inplace=True)
df_ml.dropna(subset=['publisher'], inplace=True)

# Borro los outliers y utilizo el operador ~ para decirle que seleccione todas las columnas que no cumplan esa condicion_outlier. Para este proceso los juegos con mas de $100 de precio van a ser considerados Outliers.
condicion_outliers = df_ml['price'] > 100
df_ml= df_ml[~condicion_outliers]

# Elimino las columnas que no voy a considerar en mi dataset
columnas_a_eliminar = ['specs', 'genres', 'title', 'metascore']
df_ml = df_ml.drop(columns=columnas_a_eliminar)

#Expando mi df para ver bien las diferentes tags
df_ml = df_ml.explode('tags')


# ---------------------- FIN ETL PARA ML ------------------------- #
#---------------------- INICIO MODELO ---------------------------#

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Separo las columnas publisher, tags, sentiment y anio como variables independientes y price como variable dependiente.
X = df_ml[['publisher', 'tags', 'sentiment', 'anio']]
y = df_ml['price']

# Divido el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Transformo caracteristicas categoricas utilizando codificación One-Hot para poder utilizarlas como X 
categorical_features = ['publisher', 'tags', 'sentiment', 'anio']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Combino las transformaciones en un preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Creo el modelo de Regresion Lineal
linear_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])

# Ajusto el modelo con los datos de entrenamiento
linear_model.fit(X_train, y_train)


# Realizo predicciones en el conjunto de prueba
y_pred = linear_model.predict(X_test)

# Calculo el RMSE del modelo
rmse_modelo = round(mean_squared_error(y_test, y_pred, squared=False),2)



