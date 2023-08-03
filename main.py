import pandas as pd
import numpy as np
import ast
from datetime import datetime


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


def genero(anio):
    
    df_gen_anio = df_generos[df_generos.anio == anio]
    lista_generos = []
    for _ in range (5):
        mayor = df_gen_anio.genres.describe().top
        lista_generos.append(mayor)
        df_gen_anio =  df_gen_anio[df_gen_anio['genres'] != mayor]
        
    return {anio:lista_generos}


def juegos(anio):
    df_juegos = df[df['anio'] == anio]
    return {anio: list(df_juegos.title)}
    


def specs(anio):
    df_specs_anio = df_specs[df_specs.anio == anio]
    lista_specs = []
    for _ in range (5):
        mayor = df_specs_anio.specs.describe().top
        lista_specs.append(mayor)
        df_specs_anio =  df_specs_anio[df_specs_anio['specs'] != mayor]
        
    return {anio:lista_specs}	


def earlyaccess(anio):
    df_early = df[df['anio'] == anio]
    return {anio:df_early[df_early.early_access == True].title.count()}
    

def sentiment(anio):
    df_sent = df[df['anio'] == anio]
    dicc = {}
    mayor = ''
    cantidad = 0
    for i in range (len(df_sent.sentiment.unique())):
        if df_sent.sentiment.describe().top == 'No data':
            df_sent = df_sent[df_sent['sentiment'] != 'No data']
        else:
            mayor = df_sent.sentiment.describe().top
            cantidad = df_sent.sentiment.describe().freq
            dicc[mayor] = cantidad
            df_sent = df_sent[df_sent['sentiment'] != mayor]
    
    return dicc


def metascore(anio):
    df_meta = df[df['anio'] == anio]
    dicc_mayores = {}

    df_meta_mayor = df_meta.sort_values(by="metascore", ascending=False)
    for _, fila in df_meta_mayor.iterrows():
        if len(dicc_mayores) == 5 or df_meta_mayor.metascore.count() == len(dicc_mayores):
            break
        else:
            dicc_mayores[fila['title']] = fila['metascore']
    if len(dicc_mayores) == 0:
        return "No hay valoraciones registradas este anio"
    
    return dicc_mayores


#Convierto el archivo json en una lista de python por lineas.
lineas_js = []
with open('./src/steam_games.json') as f:
    for line in f.readlines():
        lineas_js.append(ast.literal_eval(line))

df = pd.DataFrame(lineas_js)

# Relleno los nulos en 'title' con los valores de 'app_name y elimino title para evitar redundancias.

df['title'] = df.apply(lambda row: row['app_name'] if pd.isnull(row['title']) else row['title'], axis=1)
df = df.drop(columns='app_name')

#Veo los nulos y borro el de indice 74 porque tiene practicamente toda la informacion NaN
df[df['title'].isnull()]

df = df.drop(74)

df.price.unique()

df['price'] = df['price'].replace(['Free To Play', 'Free to Play', 'Free', 'Free Demo', 'Play for Free!', 'Install Now', 'Play WARMACHINE: Tactics Demo', 'Free Mod', 'Install Theme', 'Third-party', 'Play Now', 'Free HITMAN™ Holiday Pack', 'Play the Demo', 'Free to Try', 'Free Movie', 'Free to Use'], 0.00)
df['price'] = df['price'].replace(['Starting at $499.00'], 499.00)
df['price'] = df['price'].replace(['Starting at $449.00'], 449.00)


df['anio'] = df.release_date.apply(convertir_a_anio)

#Expando el dataframe original para ver los distintos generos y Creo un dataframe nuevo con las columnas Genero y Anio
df_generos = df.explode('genres')
columnas_a_mantener = ['genres', 'anio']
df_generos = df_generos.drop(columns=[col for col in df_generos.columns if col not in columnas_a_mantener])


#Expando el dataframe original para ver los distintos specs y Creo un dataframe nuevo con las columnas a Specs y Anio
df_specs = df.explode('specs')
columnas_a_mantener = ['specs', 'anio']
df_specs = df_specs.drop(columns=[col for col in df_specs.columns if col not in columnas_a_mantener])

lista_generos = df_generos.genres.unique()

#Lleno los datos NaN con un string No data en la columna sentiment para poder trabajar esa columna completa
df.sentiment.fillna("No data", inplace=True)

# Convierto los valores de string 'NA' en valores NaN con el parametro errors=coerce. Esto lo hago para poder tener solo numeros y NaN en la columna.
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')


