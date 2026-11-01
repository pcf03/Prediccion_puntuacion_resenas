import pandas as pd
import numpy as np
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import HDBSCAN
import nltk
from nltk.corpus import stopwords


# 1) Config inicial y descarga de palabras vacias (stopwords)
nltk.download("stopwords",quiet=True)
SEED = 42

# Carga de datos
df = pd.read_csv("datasets/hotelsb_df_es.csv", encoding="utf-16")
# Eliminamos posibles valores nulos por precaución
df = df.dropna(subset=["description", "rating"])
# Asigno las columnas al csv
X_texto = df["description"]
y_estrellas = df["rating"]

# 2) Separacion de datos 
# Separamos el test ANtes de estimar cualquier transformacion para evitar fugas de informacion
X_train_texto, X_test_texto, y_train, y_test = train_test_split(
    X_texto, y_estrellas, test_size=0.25, random_state=SEED
)

# 3) Vectorizacion TF-IDF 
# Transforma el texto en frequencias ponderadas, eliminando las palabras q no tienen valor semantico
vectorizador = TfidfVectorizer(stop_words=stopwords.words("spanish"))

# ajustar fit solo con entrenamiento y transformamos los dos conjuntos
X_train_tfidf = vectorizador.fit_transform(X_train_texto).toarray()
X_test_tfidf = vectorizador.transform(X_test_texto).toarray()

# 4) Reduccion de Dimensionalidad con UMAP
# Reducimos las columnas a solo 10 principales
reductor_umap = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, random_state=SEED)

# El reductor aprende solo de X_train_tfidf
X_train_umap = reductor_umap.fit_transform(X_train_tfidf)
X_test_umap = reductor_umap.transform(X_test_tfidf)

# 5) Clustering de las reseñas de entregnamiento con HDBSCAN
# extrae grupos basados en densidad sin forzar un numero K de clusteres
# minimo tiene que tener 30 hoteles para considerarse un grupo
agrupador = HDBSCAN(min_cluster_size=30)
etiquetas_clusters = agrupador.fit_predict(X_train_umap)

# evaluacion rapida del ruido generado por HDBSCAN 
ruido = (etiquetas_clusters == -1).mean()
numero_grupos = len(set(etiquetas_clusters) - {-1})
print(f"HDBSCAN detectó {numero_grupos} temáticas distintas en las descripciones.")
print(f"El {ruido:.0%} de las descripciones fueron catalogadas como ruido/sin estructura clara.")

# Exportar las matrices a un archivo comprimido para luego cargarlas
np.savez("resultados/datos_procesados.npz", 
         X_train=X_train_umap, 
         X_test=X_test_umap, 
         y_train=y_train, 
         y_test=y_test)
print("¡Archivos numéricos exportados con éxito a la carpeta resultados!")






