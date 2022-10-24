# Project 6 Data Science OpenClassrooms

## Unsupervised Learning : NLP and Clustering

Author : Oumeima EL GHARBI.

Date : October, November 2022.



### Context

Pour rendre l’expérience utilisateur des vendeurs (faciliter la mise en ligne de nouveaux articles) et des acheteurs (faciliter la recherche de produits) la plus fluide possible, et dans l'optique d'un passage à l'échelle, il devient nécessaire d'automatiser cette tâche.

Linda, Lead Data Scientist, vous demande donc d'étudier la faisabilité d'un moteur de classification des articles en différentes catégories, avec un niveau de précision suffisant.

*****

Ta mission est de réaliser une première étude de faisabilité d'un moteur de classification d'articles, basé sur une image et une description, pour l'automatisation de l'attribution de la catégorie de l'article.

Tu dois analyser le jeu de données en réalisant un prétraitement des descriptions des produits et des images, une réduction de dimension, puis un clustering. Les résultats de la réduction de dimension et du clustering seront à présenter sous la forme de graphiques en deux dimensions, et confirmés par un calcul de similarité entre les catégories réelles et les clusters. Ces résultats illustreront le fait que les caractéristiques extraites permettent de regrouper des produits de même catégorie.

**Problématique :**
Pourrais-tu nous démontrer, par cette approche de modélisation, la faisabilité de regrouper automatiquement des produits de même catégorie ?

Voici les contraintes :

Afin d’extraire les **features texte**, il sera nécessaire de mettre en œuvre :
- deux approches de type **“bag-of-words”**, **comptage simple de mots** et **Tf-idf** ;
- une approche de type **word/sentence** embedding classique avec Word2Vec (ou Glove ou FastText) ;
- une approche de type **word/sentence** embedding avec **BERT** ;
- une approche de type **word/sentence** embedding avec **USE** (Universal Sentence Encoder).


Afin d’extraire les **features image**, il sera nécessaire de mettre en œuvre :
- un algorithme de type SIFT / ORB / SURF ;
- un algorithme de type CNN Transfer Learning.


***********
We have dataset provided by Olist, the largest department store in Brazilian marketplaces, and we want to make a **Customer Segmentation**.
This is a project about **Unsupervised Learning** in which we will use **Clustering Algorithms**.

Using a RFM Segmentation, we have tried these clustering algorithms :

- Centroid-based Clustering : **K-Means**
- Hierarchical Clustering : **Agglomerative Clustering**
- Density-based Clustering : **DBSCAN**

We also computed a RFM (Recency, Frequency, Monetary) Score and tried a Segmentation based on Personae.
***********


### Dataset folder

- Create a folder **dataset**

- Create a folder **dataset/source**
- Create a folder **dataset/cleaned**

- Download the zip folder at this address and unzip it in the **source** folder :
  https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip
 


### Model folders

- Create a folder **model**

### Libraries

Install the python libraries with the same version :

```bash
pip install -r requirements.txt
```

pip freeze | findstr nltk

### Execution

```bash
run P6_01_notebook.ipynb
```

