import streamlit as st
import pandas as pd
import joblib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Titre de l'application
st.title("Clustering avec DBSCAN et visualisation t-SNE")

# Charger les modèles sauvegardés
dbscan_model = joblib.load('dbscan_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lire le CSV
    df = pd.read_csv(uploaded_file)

    # Garder uniquement les colonnes spécifiées
    columns_to_keep = ['searchTerms', 'title', 'snippet', 'link']

    if all(col in df.columns for col in columns_to_keep):
        df = df[columns_to_keep]
        #st.write("Aperçu des colonnes sélectionnées :", df.head())

        # Combiner les colonnes pour le texte complet (si nécessaire)
        df['text'] = df['searchTerms'] + " " + df['title'] + " " + df['snippet']

        # Appliquer la vectorisation TF-IDF sur le texte
        X_new_tfidf = tfidf_vectorizer.transform(df['text'])

        # Appliquer le modèle DBSCAN pré-entraîné
        clusters = dbscan_model.fit_predict(X_new_tfidf)

        # Ajouter les clusters à la DataFrame
        df['Cluster'] = clusters

        # Afficher les clusters dans la table
        st.subheader("Données avec les clusters assignés")
        st.write(df)

        # Réduire la dimensionnalité avec t-SNE pour la visualisation
        st.subheader("Visualisation des clusters avec t-SNE")

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_new_tfidf.toarray())  # t-SNE exige un array dense

        # Créer un DataFrame avec les résultats de t-SNE
        df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
        df_tsne['Cluster'] = clusters

        # Visualiser les clusters avec t-SNE
        fig, ax = plt.subplots()
        sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', data=df_tsne, palette="Set1", ax=ax, legend="full")
        plt.title("Visualisation des clusters avec t-SNE")
        st.pyplot(fig)

    else:
        st.error(f"Les colonnes {columns_to_keep} sont manquantes dans votre fichier CSV.")
