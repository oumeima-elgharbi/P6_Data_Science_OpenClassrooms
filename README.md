# Project 6 Data Science OpenClassrooms

## About NLP, SIFT, CNN and Transfer Learning, Clustering

Author : Oumeima EL GHARBI.

Date : October, November 2022.

### Context

The company "Place du Marché" for which we are working for wants to see if we can automatically classify new products
that vendors want to add to the marketplace.

Thus, by using Natural Language Processing and SIF, CNN algorithms we want to see if an automatic classification is
doable.

#### Text

To extract the **textual features**, we use :

- **“bag-of-words”** : **n grams** and **Tf-idf**
- **word/sentence** embedding with Word2Vec
- **word/sentence** embedding with **BERT**
- **word/sentence** embedding with **USE** (Universal Sentence Encoder)

#### Images

To extract the **images features**, we use :

- SIFT algorithm (Scale-invariant feature transform)
- CNN Transfer Learning algorithm

### Dataset folder

- Create a folder **dataset**

- Create a folder **dataset/source**
- Create a folder **dataset/images**

- Download the zip folder at this address and unzip it in the **source** folder :
  https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip
  Put the csv file in **dataset/source** and rename the folder **Images** as **images**

- Create a folder **dataset/cleaned**

### Libraries

Install the python libraries with the same version :

```bash
pip install -r requirements.txt
```

python version 3.9

#### Information about the requirements:

With this command below, we got the requirements file for this project :

```bash
pipreqs
```

If you encounter problem with the librairies in the requirements.txt :

C:\ProgramData\Anaconda3\python.exe -m pip install --upgrade --force-reinstall opencv-python

pip install "tensorflow>=2.0.0"
pip install --upgrade tensorflow-hub

conda uninstall tokenizers, transformers
pip install transformers

### Execution

First, run the EDA notebook.

#### Exploratory Data Analysis

```bash
run P6_01_cleaning_exploration.ipynb
```

The notebooks about NLP, SIFT and CNN are independent. So, you can run them in the order you want.

#### NLP

```bash
run P6_02_preprocess_NLP.ipynb
run P6_03_NLP.ipynb
```

#### SIFT

```bash
run P6_02_SIFT.ipynb
```

#### CNN

```bash
run P6_02_CNN.ipynb
```
