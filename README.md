# Datahon 2023- ProyectoSeguroGalicia
## Project
Participants should develop a model that identifies the ideal combo of coverage and coverage and assistance for PYMES
and entrepreneurs according to their particular characteristics based on a set of data provided by Galicia Seguros particular
characteristics based on a set of data provided by Galicia Seguros plus additional information
from external sources that the participants themselves can provide, by current data laws.

## Problem Statement
To address this challenge, we explored two clustering models: K Modes and DBSCAN.
We use DBSCAN because of its ability to identify regions of high density in numerical datasets, 
which is useful in our task. In addition, for the K Modes model, we take advantage of its ability to identify 
trends and centroids using modes in categorical data observations.

![](/DATOS.png?raw=true "")

## Solution proposed 

Since our data have different types (numerical and categorical), we apply specific processing strategies. 
For the K Modes model, we transform the numerical data into ranges to increase the number of observations. 
Then, we used Scikit-learn and the LabelEncoder class to convert the categorical data into numerical data. 
For DBSCAN testing, we create a Pipeline by combining the OneHotEncoder and StandardScaler classes to ensure that all data is in the same format. 
This allows us to take categorical data to numeric data and maintain the same scale for the numeric data, preserving the variance unique to this data type.
We use the ColumnTransformer class to perform these data transformations separately.

![](/LSTM image.png?raw=true "")

Once we select a set of clusters, we can characterize specific groups of customers. In addition, we consider the use of an LSTM-type recurrent neural network (RNN) to predict customer behavior
based on sequential data and the individual characteristics of each customer.



## Configuración del Entorno

### Crear un nuevo entorno virtual

Primero, see debe tener Python 3.x instalado en sistema. Luego, crea un nuevo entorno virtual:

```bash
python -m venv <environment_name>
```

### Activar el entorno virtual (Linux/Mac)

```bash
source <environment_name>/bin/activate
```

### Activar el entorno virtual (Windows)

```bash
<environment_name>\Scripts\activate
```

## Instalar Dependencias

Una vez que tengas el entorno virtual activado, puedes instalar las dependencias del proyecto desde el archivo `requirements.txt` usando pip:

```bash
pip install -r requirements.txt
```
