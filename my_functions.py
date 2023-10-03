import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
import regex as re
import io

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

#Funciones Lectura

def download_file(container_client, archivo_a_cargar ):

    blob_client = container_client.get_blob_client(archivo_a_cargar)

    blob_data = blob_client.download_blob()
    archivo_bytes = blob_data.readall()

    data = pd.read_csv(io.BytesIO(archivo_bytes), delimiter=';' )

    return data
    pass



def read_files(path=None):
    if path is None:
        files = [file for file in os.listdir() if '.csv' in file]
    else:
         files = [os.path.join(path, file) for file in os.listdir(path) if '.csv' in file]
    
    data_dict = {}
    for file in files:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        data_dict[filename] = df
    
    return data_dict

    pass

def join_csv(path, out_file):
    if path is None:
        files = [file for file in os.listdir() if '.csv' in file]
    else:
        files = [os.path.join(path, file) for file in os.listdir(path) if '.csv' in file]

    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    
    if out_file is not None:
        if path is None:
            df.to_csv(out_file)
        else:
            df.to_csv(os.path.join(path, out_file))
        
    return df

    pass

def join_parquet(path, out_file):
    if path is None:
        files = [file for file in os.listdir() if '.csv' in file]
    else:
        files = [os.path.join(path, file) for file in os.listdir(path) if '.csv' in file]

    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    
    if out_file is not None:
        if path is None:
            df.to_parquet(out_file)
        else:
            df.to_parquet(os.path.join(path, out_file))
        
    return df

    pass

def busqueda_numericas(data):
    columnas_num = []
    for column in data.columns:
        if data[column].dtype == 'int64' or data[column].dtype == 'float64':
            columnas_num.append(column)
    
    data_num = data[columnas_num]
    return data_num

    pass

def busqueda_categorica(data):
    columnas_cat = []
    for column in data.columns:
        if data[column].dtype == 'object':
            columnas_cat.append(column)
    
    data_cat = data[columnas_cat]
    return data_cat
    pass

#Funciones de Limpieza

def Limpieza_personas(data, data_pyme):

    merged_data = data.merge(data_pyme[['KEY_CLIENT', 'PROVINCIA']], on='KEY_CLIENT', how='left')
    data['PROVINCIA'] = merged_data['PROVINCIA_x'].fillna(merged_data['PROVINCIA_y'])
    data.replace({'Capital Federal': 'Ciudad Autonoma Buenos Aires'}, inplace=True)

    Dic2 = {'Ciudad Autonoma Buenos Aires': 'Ciudad Autónoma de Buenos Aires', 'Tucuman':'Tucumán', 'Cordoba':'Córdoba', 'Entre Rios': 'Entre Ríos', 'Neuquen': 'Neuquén', 'Rio Negro':'Río Negro'}
    data = data.replace(Dic2)
   
    return data
    pass

def Limpieza_datos_registropyme(data):

    test = data.groupby(['CLAE6', 'Sector']).size().reset_index(name='Cantidad')
    test = test.sort_values(by='Cantidad', ascending=False)
    test = test.drop_duplicates(subset='CLAE6', keep='first')
    data_sector = test.iloc[:,:2]
    data_sector = data_sector.rename(columns={'CLAE6':'COD_ACT_AFIP'})

    return data_sector
    pass

def Limpieza_datos_coberturas(data):
    data.drop(['Unnamed: 0'],axis = 1, inplace=True)
    data['DESCRIPCION_COBERTURA'] = data['DESCRIPCION_COBERTURA'].str.replace(r'\xa0', ' ')
    data['DESCRIPCION_COBERTURA'] = data['DESCRIPCION_COBERTURA'].str.replace(r'\s{2,}', ' ', regex=True).str.strip()
    df_encoded = pd.get_dummies(data, prefix = 'Cobertura', columns=['DESCRIPCION_COBERTURA'])
    df_agrupado = df_encoded.groupby(['PERIODO', 'KEY_CLIENT']).sum().reset_index()
    df_agrupado['PERIODO'] = pd.to_datetime(df_agrupado['PERIODO'], format='%Y%m').dt.strftime("%Y-%m")
    df_agrupado
    return df_agrupado
    pass

def Limpieza_datos_risk(data):

    data = data.rename(columns={'Provincia': 'PROVINCIA'})

    for columna in data.iloc[:, 1:].columns:
        data[columna] = data[columna].str.replace(",", ".").astype(float)


    Dic = {' Ciudad Autonoma Buenos Aires':'Ciudad Autónoma de Buenos Aires', 'Buenos Aires': 'Buenos Aires', ' San Luis': 'San Luis', ' Chaco':'Chaco', ' Chubut':'Chubut', ' Entre Rios':'Entre Ríos', ' La Pampa': 'La Pampa', 'Formosa': 'Formosa', ' Cordoba': 'Córdoba',
       ' Catamarca':'Catamarca', ' Corrientes': 'Corrientes', ' Jujuy':'Jujuy', ' La Rioja': 'La Rioja', ' Mendoza':'Mendoza', ' Rio Negro':'Río Negro', ' Salta':'Salta', ' San Juan': 'San Juan', ' Misiones': 'Misiones', ' Neuquen': 'Neuquén',
       ' Santiago del Estero': 'Santiago del Estero', ' Tucuman': 'Tucumán', ' Tierra del Fuego': 'Tierra del Fuego', ' Santa Cruz': 'Santa Cruz', ' Santa Fe': 'Santa Fe'}
    
    data = data.replace(Dic)
    

    return data
    pass

def Limpieza_datos_delincuencia(data):

    data = data.rename(columns={'provincia_nombre': 'PROVINCIA', 'anio': 'Año'})
    Dic1 = {'Tierra del Fuego, Antártida e Islas del Atlántico Sur': 'Tierra del Fuego'}
    data = data.replace(Dic1)
    Mask1 = data['Año'] == 2022
    data = data.loc[Mask1]
    pivot_data = data.pivot(index='PROVINCIA', columns='codigo_delito_snic_nombre', values='cantidad_hechos')
    pivot_data.reset_index(inplace=True)
    pivot_data = pivot_data.rename_axis('', axis=1)

    columnas= ['PROVINCIA', 'Homicidios dolosos', 'Hurtos',
       'Lesiones culposas en Accidentes Viales', 'Lesiones dolosas',
       'Muertes en Accidentes Viales','Otros delitos contra la propiedad',
       'Otros delitos contra las personas',
       'Robos (excluye los agravados por el resultado de lesiones y/o muertes)',
       'Robos agravados por el resultado de lesiones y/o muertes ', 'Tentativas de hurto',
       'Tentativas de robo (excluye las agravadas por el res. de lesiones y/o muerte)',
       'Tentativas de robo agravado por el resultado de lesiones y/o muertes']

    delincuencia_filtrado = pivot_data[columnas]

    return delincuencia_filtrado
    pass

def Limpieza_siniestros(data):
    
    Dic1 = {'HA-TensiÃ³n':'HA-Tension', 'HN-Granizo': 'HN-Granizo', 'HA-Rotura Otros Bienes o ArtÃ\xadc':'HA-Rotura Otros Bienes o Art',
       'HD - Robo con Efraccion': 'HD-Robo con Efraccion', 'HA - Caida de Objeto':'HA-Caida de Objeto' , 'HD-Hurto':'HD-Hurto',
       'HN-Rayo':'HN-Rayo', 'HN - Rayo':'HN-Rayo', 'HA - TensiÃ³n':'HA-Tension',
       'HA - DaÃ±o por Humo y HollÃ­n':'HA-Daño por Humo y Hollin', 'HA-CaÃ\xadda de objeto':'HA-Caida de Objeto',
       'HA-Rotura de Cristales':'HA-Rotura de Cristales', 'HD-Robo con EfracciÃ³n':'HD-Robo con Efraccion',
       'HA-Cortocircuito':'HA-Cortocircuito', 'Robo de celular':'HD-Daños por Robo o Tentativa',
       'Robo Equipos ElectrÃ³nicos':'HD-Daños por Robo o Tentativa', 'HA-Incendio':'HA-Incendio', 'HN-Temporal':'HN-Temporal',
       'Robo en Cajero':'HD-Daños por Robo o Tentativa', 'Robo de Celular':'HD-Daños por Robo o Tentativa', 'HA - Incendio':'HA-Incendio',
       'HA - Rotura de Cristales':'HA-Rotura de Cristales', 'HD - Robo por Asalto':'HD-Robo por Asalto',
       'HA - Derrumbe':'HA-Derrumbe', 'Robo VÃ\xada Publica':'HD-Daños por Robo o Tentativa',
       'HD-DaÃ±os por Robo o Tentativa':'HD-Daños por Robo o Tentativa', 'HA - DaÃ±o por Agua/Humedad':'HA-Daños por Agua/Humedad',
       'HA-DaÃ±o por agua/Humedad':'HA-Daños por agua/Humedad', 'PÃ©rdida Alimentos':'HA-Rotura Otros Bienes o Art',
       'Fallecimiento Enf. Mascota':'Otros accidentes', 'RC Responsabilidad civil':'HA-Impacto Vehiculo Terrestre',
       'HA - CaÃ\xadda de objeto':'HA-Caida de Objeto', 'HA - Rotura Otros Bienes o Art':'HA-Rotura Otros Bienes o Art',
       'Robo Bienes':'HD-Daños por Robo o Tentativa', 'HD - DaÃ±os Robo o Tentativa':'HD-Daños por Robo o Tentativa',
       'HA - Impacto vehÃ\xadculo terr':'HA-Impacto Vehiculo Terrestre', 'RC-Responsabilidad Civil':'HA-Impacto Vehiculo Terrestre',
       'HN - Temporal':'HN-Temporal', 'HA - Impacto Vehiculo Terr':'HA-Impacto Vehiculo Terrestre',
       'DaÃ±o por Accidente':'Otros accidentes', 'HD-Robo por Arrebato':'HD-Robo por Arrebato',
       'Estancia en Residencia Mascota':'Otros accidentes', 'HA - ExplosiÃ³n':'HA-Explosion', 'HD - Hurto':'HD-Hurto',
       'HA - ColisiÃ³n':'HA-Colision', 'DaÃ±o Accidental':'Otros accidentes', 'Otros accidentes':'Otros accidentes',
       'Gastos Asistencia Veterinaria':'Otros accidentes', 'HA - Impacto de aeronave':'HA-Impacto de aeronave',
       'HN - HuracÃ¡n':'HN-Huracan', 'Otras enfermedades':'Otros accidentes', 'HN - Granizo':'HN-Granizo',
       'Gastos por Sacrificio':'Otros accidentes', 'Gastos por ExtravÃ\xado':'HA-Rotura Otros Bienes o Art',
       'HD - Robo por Arrebato':'HD-Robo por Arrebato', 'Fallecimiento Acc. Mascota':'Otros accidentes'}
    

    data = data.replace(Dic1)

    fecha_columns = ['FECHA_DENUNCIA', 'FECHA_OCURRENCIA', 'FECHA_CIERRE']
    for col in fecha_columns:
        data[col] = pd.to_datetime(data[col], format='%Y-%m-%d')

    data['FECHA_DENUNCIA'] = data['FECHA_DENUNCIA'].dt.strftime('%Y-%m-%d')

    data['diferencia'] = (data['FECHA_CIERRE'] - data['FECHA_OCURRENCIA']).dt.days

    data['DIFF_DENUNCIA_CIERRE'].fillna(data['diferencia'], inplace=True)

    data['DIFF_DENUNCIA_CIERRE'] = data['DIFF_DENUNCIA_CIERRE'].apply(lambda x: round(x) if pd.notna(x) else x)

    mask = data['DIFF_DENUNCIA_PAGO'].isna()

    data.loc[mask, 'FLG_RECHAZO'] = 1

    data['FLG_RECHAZO'] = data.apply(lambda row: 1 if row['FLG_PAGO'] == 0 else row['FLG_RECHAZO'], axis=1)

    data['DIFF_DENUNCIA_PAGO'].fillna(0, inplace=True)

    bins = [-np.inf, 1, 20, 30, 100]
    labels = ['Inmediato', 'Menor a 30 días', 'Mayor a 30 días', 'Demorado']
    data['Categoría días pagos'] = np.where(data['FLG_PAGO'] == 1, pd.cut(data['DIFF_DENUNCIA_PAGO'], bins=bins, labels=labels), 'No pago')
    
    return data
    pass

def Limpieza_tenencias(data):
    columnas = data.iloc[:,:42]

    data['Total_coberturas'] = columnas.sum(axis=1)
    data['Cantidad_real_vigentes'] = data.apply(lambda row: row['CANT_POL_VIGENTES'] <= (row['FLG_SIP_VIGENTE'] + row['Total_coberturas']), axis=1)
    name_columns = ['CANT_POL_VIGENTES', 'FLG_SIP_VIGENTE', 'Total_coberturas']  
    condicion0 = (data['FLG_SIP_VIGENTE'] == 0) & (data['CANT_POL_VIGENTES'] >= (data['FLG_SIP_VIGENTE'] + data['Total_coberturas']))
    data['CANT_POL_VIGENTES'] = np.where(condicion0, data['Total_coberturas'], data['CANT_POL_VIGENTES'])
    condicion1 = (data['FLG_SIP_VIGENTE'] == 0) & (data['CANT_POL_VIGENTES'] >= (data['FLG_SIP_VIGENTE'] + data['Total_coberturas']))
    data['CANT_POL_VIGENTES'] = np.where(condicion1, data['Total_coberturas'], data['CANT_POL_VIGENTES'])
    condicion2 = (data['FLG_SIP_VIGENTE'] == 1) & (data['CANT_POL_VIGENTES'] >= (data['FLG_SIP_VIGENTE'] + data['Total_coberturas']))
    data['CANT_POL_VIGENTES'] = np.where(condicion2, data['Total_coberturas']+data['FLG_SIP_VIGENTE'], data['CANT_POL_VIGENTES'])

    columnas_core1 = data.filter(like='_CORE_1')
    suma_core1 = columnas_core1.sum(axis=1)

    columnas_core2 = data.filter(like='_CORE_2')
    suma_core2 = columnas_core2.sum(axis=1)

    data['Suma_core1'] = suma_core1
    data['Suma_core2'] = suma_core2

    column_names = data.columns.tolist()

    column_names.insert(43, column_names.pop(42))
    

    return data
    pass

def Limpieza_contactos(data):
    data['FECHA_GESTION'] = pd.to_datetime(data['FECHA_GESTION'], format='%Y-%m-%d').dt.strftime("%Y-%m-%d")
    data['DESC_GEST'] = data.apply(lambda row: 'GESTIONES' if pd.isna(row['DESC_GEST']) and row['FLG_WPP'] == 0 else row['DESC_GEST'], axis=1)
    
    mask_0 = pd.isna(data['DESC_GEST'])
    mask_1 = data['FLG_WPP'] == 1
    data_wapp = data[mask_0 & mask_1]
    distribucion_reemplazo = {
    'SOLICITA POLIZA': 0.50,
    'POST-VENTA-ASESORAMIENTO': 0.50,
    }
    valores_de_reemplazo = np.random.choice(list(distribucion_reemplazo.keys()), size=len(data_wapp), p=list(distribucion_reemplazo.values()))
    data_wapp['DESC_GEST']= valores_de_reemplazo

    indices_comunes = data.index.intersection(data_wapp.index)
    data.loc[indices_comunes] = data_wapp.loc[indices_comunes]

    data['cant_contactos_tipo_DESC_GEST'] = data.groupby('KEY_CLIENT')['DESC_GEST'].transform('count')
    return data
    pass

def Limpieza_certificados(data):
    data_columns = ['ANTIGUEDAD_CLIENTE', 'ANTIGUEDAD_POLIZA_MAS_RECIENTE']
    
    for column in data_columns:
        if data[column].dtype == 'int64':
            fecha_referencia = datetime.date(2022, 12, 31)
            fecha_referencia = pd.to_datetime(fecha_referencia)
            data['ANTIGUEDAD_CLIENTE'] = pd.to_timedelta(data['ANTIGUEDAD_CLIENTE'], unit='D')
            data['fecha_antiguedad_cliente'] = fecha_referencia - data['ANTIGUEDAD_CLIENTE']
            data['ANTIGUEDAD_POLIZA_MAS_RECIENTE'] = pd.to_timedelta(data['ANTIGUEDAD_POLIZA_MAS_RECIENTE'], unit='D')
            data['fecha_antiguedad_ulit_poliza'] = fecha_referencia - data['ANTIGUEDAD_POLIZA_MAS_RECIENTE']
        else:
            None

    data['fecha_sin_interaccion'] = data.apply(lambda row: row['ANTIGUEDAD_CLIENTE'] if row['ANTIGUEDAD_CLIENTE'] == row['ANTIGUEDAD_POLIZA_MAS_RECIENTE'] else row['ANTIGUEDAD_CLIENTE'] - row['ANTIGUEDAD_POLIZA_MAS_RECIENTE'], axis=1)
    
    return data
    pass

def Limpieza_Pyme(data, data2, data3):
    data.rename(columns= {'LOCALIDAD': 'CIUDAD'}, inplace=True)
    data['PROVINCIA'] = data.apply(lambda row: 'Ciudad Autónoma de Buenos Aires' if row['CIUDAD'] == 'CABA' else row['PROVINCIA'], axis=1)
    data['PROVINCIA'] = data.apply(lambda row: 'Tierra del Fuego' if row['PROVINCIA'] == 'Ushuaia' else row['PROVINCIA'], axis=1)

    columnas = ['Q_EMPLEADOS',	'PAGO_SUELDO_CCSS',	'PERFIL_INVERSOR',	'SCORING_CREDITICIO',	'KEY_CLIENT', 'Unnamed: 12' ]

    Lista_filaNew = []
    for index, row in data.iterrows():
        valor = row['Unnamed: 12']
        if isinstance(valor, str) and not valor.isnumeric():
            Lista_filaNew.append(index)
            print(f"Fila {index}, Columna {'Unnamed: 12'}: Valor no numérico = {valor}")

    for fila in Lista_filaNew:
        valor_C = data.at[fila, "Q_EMPLEADOS"]
        valor_D = data.at[fila, "PAGO_SUELDO_CCSS"]
        valor_E = data.at[fila, "PERFIL_INVERSOR"]
        valor_F = data.at[fila, "SCORING_CREDITICIO"]
        valor_G = data.at[fila, "KEY_CLIENT"]
        Valor_H = data.at[fila, "Unnamed: 12"]
        
        data.at[fila, "RENTABILIDAD"] = valor_C
        data.at[fila, "Q_EMPLEADOS"] = valor_D
        data.at[fila, "PAGO_SUELDO_CCSS"] = valor_E
        data.at[fila, "PERFIL_INVERSOR"] = valor_F 
        data.at[fila, "SCORING_CREDITICIO"] = valor_G
        data.at[fila, "KEY_CLIENT"] = Valor_H

    data.at[fila, columnas[-1]] = None

    data['Q_EMPLEADOS'] = data['Q_EMPLEADOS'].astype(int)
    data['PAGO_SUELDO_CCSS'] = data['PAGO_SUELDO_CCSS'].str.replace('[^\d.]', '', regex=True).astype(float)
    data['SCORING_CREDITICIO'] = data['SCORING_CREDITICIO'].str.replace('.', '').astype(int)
    data.drop(['id'], axis=1, inplace=True)
    data.drop(["Unnamed: 12"], axis=1, inplace=True) 

      
    data_sector = Limpieza_datos_registropyme(data2)

    Resultado_Pyme_Sector = data.merge(data_sector, on='COD_ACT_AFIP', how='left')

    Mask_0 = Resultado_Pyme_Sector['Q_EMPLEADOS'] > 0.0
    datos_Limpios_0 = Resultado_Pyme_Sector.loc[Mask_0]
    datos_Limpios_0['PAGO_SUELDO_CCSS'] = datos_Limpios_0['PAGO_SUELDO_CCSS'].abs() 
    Mask_02 = datos_Limpios_0['PAGO_SUELDO_CCSS'] > 0.0
    datos_Limpios_0 = datos_Limpios_0.loc[Mask_02]

    datos_Limpios_0 = datos_Limpios_0.reset_index(drop=True)

    datos_Limpios_0.dropna(subset=['DESC_ACT_AFIP'], inplace=True)

    merged_data = datos_Limpios_0.merge(data3[['KEY_CLIENT', 'PROVINCIA']], on='KEY_CLIENT', how='left')
    datos_Limpios_0['PROVINCIA'] = datos_Limpios_0['PROVINCIA'].fillna(merged_data['PROVINCIA_y'])

    distribucion_conjunta = datos_Limpios_0.groupby(['PERFIL_INVERSOR', 'DESC_SEGMENTO']).size().reset_index(name='Cantidad')
    total_cantidad = distribucion_conjunta['Cantidad'].sum()
    distribucion_conjunta['Porcentaje'] = distribucion_conjunta['Cantidad'] / total_cantidad

    distribuciones_por_segmento = {}

    distribuciones_por_segmento['PYME Micro'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'PYME Micro']
    distribuciones_por_segmento['PYME PES'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'PYME PES']
    distribuciones_por_segmento['Pyme'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'Pyme']
    distribuciones_por_segmento['NEGOCIOS Y PROFESIONALES GOLD'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'NEGOCIOS Y PROFESIONALES GOLD']
    distribuciones_por_segmento['NEGOCIOS Y PROFESIONALES'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'NEGOCIOS Y PROFESIONALES']

    filas_nans = datos_Limpios_0[datos_Limpios_0['PERFIL_INVERSOR'].isnull()]

    for index, row in filas_nans.iterrows():
        servicio_actual = row['DESC_SEGMENTO']
        distribucion_actual = distribuciones_por_segmento.get(servicio_actual)
        if distribucion_actual is not None and not distribucion_actual.empty:
            probabilidades = distribucion_actual['Porcentaje']
            
            if not np.isclose(probabilidades.sum(), 1.0):
                probabilidades = probabilidades / probabilidades.sum()

            fila_imputada = np.random.choice(
                distribucion_actual.index,
                p=probabilidades
            )
            perfil_inversor_imputado = distribucion_actual.loc[fila_imputada, 'PERFIL_INVERSOR']
            datos_Limpios_0.at[index, 'PERFIL_INVERSOR'] = perfil_inversor_imputado

    return datos_Limpios_0
    pass

def Limpieza_Pyme_Categorica(data, data2, data3):
    data.rename(columns= {'LOCALIDAD': 'CIUDAD'}, inplace=True)
    data['PROVINCIA'] = data.apply(lambda row: 'Ciudad Autónoma de Buenos Aires' if row['CIUDAD'] == 'CABA' else row['PROVINCIA'], axis=1)
    data['PROVINCIA'] = data.apply(lambda row: 'Tierra del Fuego' if row['PROVINCIA'] == 'Ushuaia' else row['PROVINCIA'], axis=1)

    columnas = ['Q_EMPLEADOS',	'PAGO_SUELDO_CCSS',	'PERFIL_INVERSOR',	'SCORING_CREDITICIO',	'KEY_CLIENT', 'Unnamed: 12' ]

    Lista_filaNew = []
    for index, row in data.iterrows():
        valor = row['Unnamed: 12']
        if isinstance(valor, str) and not valor.isnumeric():
            Lista_filaNew.append(index)
            print(f"Fila {index}, Columna {'Unnamed: 12'}: Valor no numérico = {valor}")

    for fila in Lista_filaNew:
        valor_C = data.at[fila, "Q_EMPLEADOS"]
        valor_D = data.at[fila, "PAGO_SUELDO_CCSS"]
        valor_E = data.at[fila, "PERFIL_INVERSOR"]
        valor_F = data.at[fila, "SCORING_CREDITICIO"]
        valor_G = data.at[fila, "KEY_CLIENT"]
        Valor_H = data.at[fila, "Unnamed: 12"]
        
        data.at[fila, "RENTABILIDAD"] = valor_C
        data.at[fila, "Q_EMPLEADOS"] = valor_D
        data.at[fila, "PAGO_SUELDO_CCSS"] = valor_E
        data.at[fila, "PERFIL_INVERSOR"] = valor_F 
        data.at[fila, "SCORING_CREDITICIO"] = valor_G
        data.at[fila, "KEY_CLIENT"] = Valor_H

    data.at[fila, columnas[-1]] = None

    data['Q_EMPLEADOS'] = data['Q_EMPLEADOS'].astype(int)
    data['PAGO_SUELDO_CCSS'] = data['PAGO_SUELDO_CCSS'].str.replace('[^\d.]', '', regex=True).astype(float)
    data['SCORING_CREDITICIO'] = data['SCORING_CREDITICIO'].str.replace('.', '').astype(int)
    data.drop(['id'], axis=1, inplace=True)
    data.drop(["Unnamed: 12"], axis=1, inplace=True) 
      
    data_sector = Limpieza_datos_registropyme(data2)

    Resultado_Pyme_Sector = data.merge(data_sector, on='COD_ACT_AFIP', how='left')

    Mask_0 = Resultado_Pyme_Sector['Q_EMPLEADOS'] >= 0.0
    datos_Limpios_0 = Resultado_Pyme_Sector.loc[Mask_0]
    datos_Limpios_0['PAGO_SUELDO_CCSS'] = datos_Limpios_0['PAGO_SUELDO_CCSS'].abs() 
    Mask_02 = datos_Limpios_0['PAGO_SUELDO_CCSS'] >= 0.0
    datos_Limpios_0 = datos_Limpios_0.loc[Mask_02]

    datos_Limpios_0 = datos_Limpios_0.reset_index(drop=True)

    datos_Limpios_0.dropna(subset=['DESC_ACT_AFIP'], inplace=True)

    merged_data = datos_Limpios_0.merge(data3[['KEY_CLIENT', 'PROVINCIA']], on='KEY_CLIENT', how='left')
    datos_Limpios_0['PROVINCIA'] = datos_Limpios_0['PROVINCIA'].fillna(merged_data['PROVINCIA_y'])
    
    distribucion_conjunta = datos_Limpios_0.groupby(['PERFIL_INVERSOR', 'DESC_SEGMENTO']).size().reset_index(name='Cantidad')
    total_cantidad = distribucion_conjunta['Cantidad'].sum()
    distribucion_conjunta['Porcentaje'] = distribucion_conjunta['Cantidad'] / total_cantidad

    distribuciones_por_segmento = {}

    distribuciones_por_segmento['PYME Micro'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'PYME Micro']
    distribuciones_por_segmento['PYME PES'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'PYME PES']
    distribuciones_por_segmento['Pyme'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'Pyme']
    distribuciones_por_segmento['NEGOCIOS Y PROFESIONALES GOLD'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'NEGOCIOS Y PROFESIONALES GOLD']
    distribuciones_por_segmento['NEGOCIOS Y PROFESIONALES'] = distribucion_conjunta[distribucion_conjunta['DESC_SEGMENTO'] == 'NEGOCIOS Y PROFESIONALES']

    filas_nans = datos_Limpios_0[datos_Limpios_0['PERFIL_INVERSOR'].isnull()]

    for index, row in filas_nans.iterrows():
        servicio_actual = row['DESC_SEGMENTO']
        distribucion_actual = distribuciones_por_segmento.get(servicio_actual)
        if distribucion_actual is not None and not distribucion_actual.empty:
            probabilidades = distribucion_actual['Porcentaje']
            
            if not np.isclose(probabilidades.sum(), 1.0):
                probabilidades = probabilidades / probabilidades.sum()

            fila_imputada = np.random.choice(
                distribucion_actual.index,
                p=probabilidades
            )
            perfil_inversor_imputado = distribucion_actual.loc[fila_imputada, 'PERFIL_INVERSOR']
            datos_Limpios_0.at[index, 'PERFIL_INVERSOR'] = perfil_inversor_imputado

    return datos_Limpios_0
    pass

#Funciones de Graficos

def histo_graph(data, x, y,color,color_discrete_map):
    fig = px.histogram(data, x=x, y=y,
                color=color, barmode='group',
                histfunc='avg',
                height=400, color_discrete_sequence= color_discrete_map)

    fig.update_layout(width=800, height=600)

    return fig
    pass

def linea_tendencia(data, y, x, group, title):
    fig = px.line(data, y=y, x=x, color=group, line_group=group,
              line_shape="spline", render_mode="svg",
             color_discrete_sequence=px.colors.qualitative.Dark2,
             title=title)

    fig.update_layout(legend_traceorder="reversed")
    fig.update_layout(width=1000, height=600)

    return fig
    pass

def scatter_plot(data,x, y, group,title):

    colores_personalizados = px.colors.qualitative.Dark2[::-1] 

    fig = px.scatter(data, x=x, y=y, color=group,
             color_discrete_sequence=colores_personalizados,
             title=title
            )

    fig.update_layout(width=1000, height=600)
    
    return fig 
    pass

def count_plots(data, hue, x, title):
    fig, axs = plt.subplots(nrows=1, ncols=1,figsize=(10,6))

    sns.countplot(x=x, hue=hue,data=data, ax=axs, palette="YlOrBr")
    sns.color_palette(palette="YlOrBr", n_colors=None, desat=None, as_cmap=False)

    axs.set_xticklabels(axs.get_xticklabels(), rotation = 90)

    plt.legend(title=title, fontsize='small', title_fontsize='12', loc='upper right')

    fig.tight_layout()

    plt.show()

    return
    pass

def barra_plot(data,x,y, title,labels,xlabel,ylabel):
    fig = px.bar(data, x=x, y=y,
                title=title,
                labels=labels,
                height=400, color_discrete_sequence=px.colors.sequential.Oranges)

    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)

    fig.update_layout(width=1000, height=600)

    return fig
    pass

def pie_graph(data, values, name, title):
    fig = px.pie(data, values=values, names=name, title=title,
                color_discrete_sequence=px.colors.sequential.Oranges)

    fig.update_traces(textposition='inside', textinfo='percent+label')

    fig.update_layout(width=1000, height=600)

    return fig
    pass

def pair_plot(data, hue):
   
    sns.pairplot(data, hue=hue, palette = "YlOrBr")

    return
    pass

def scatter_size(data, x, hue, y, size):
    
    f, ax = plt.subplots(figsize=(12, 10))

    sns.despine(f, left=True, bottom=True)

    sns.scatterplot(x=x, y=y,
                    hue=hue,
                    palette="YlOrBr",
                    sizes=(5, 100), linewidth=0, size=size,
                    data=data, ax=ax)
    plt.show()

    return
    pass

#Funciones Regex

def reemplazar_valor_0(valor):
    dic_servicios = {'Servicios':{'Enseqanza', 'Transporte', 'Reparacisn', 'Fabricacisn', 'Reparacion',
                                   'Confeccisn', 'Servicio', 'Procesamiento', 'Elaboracisn', 'Limpieza'},
                                     'Comercio': {'Enseres'}, 'Industria': {'Productos', 'Confeccion', 'Fabricacion', 'Produccisn'}, 
                                     'Construccion': {'Construccisn', 'Terminacisn'}, 'Agropecuario': {'Cultivo','Ganado'}, 'Mineria':{}}     
    for categoria, palabras_clave in dic_servicios.items():
        if valor in palabras_clave:
            return categoria
        
    return valor
    pass

def reemplazar_valor(valor):
    Dic = {'CIUDAD AUTONOMA DE BUENOS AIRES': {'CIUDAD AUTONOMA BUENOS AI', 'CIUDAD AUTÓNOMA DE BUENOS AIRES', 'CIUDAD AUTONOMA BUENOS AIRES', 'CIUDAD AUTON. BS.AS.', 'CABALLITO-CAPITAL FE',
                                                '(1425) - CAPITAL FEDERAL', 'CIUDAD AUTONOMA BS AS', 'SAAVEDRA-CAPITAL FED', 'BOEDO-CAPITAL FEDERA', 'CIUDAD AUTONOMA DE BUENOS AIRE',
                                                  'BARRACAS-CAPITAL FED', '(1406) - CAPITAL FEDERAL', 'CAPITAL FED'} }
    for categoria, palabras_clave in Dic.items():
        if valor in palabras_clave:
            return categoria
        
    return valor
    pass

def mapear_ciudad_a_capital(row):
    Provincias_capitales = {'Buenos Aires': 'La Plata',
    'Catamarca': 'San Fernando del Valle de Catamarca',
    'Chaco': 'Resistencia',
    'Chubut': 'Rawson',
    'Córdoba': 'Córdoba',
    'Corrientes': 'Corrientes',
    'Entre Ríos': 'Paraná',
    'Formosa': 'Formosa',
    'Jujuy': 'San Salvador de Jujuy',
    'La Pampa': 'Santa Rosa',
    'La Rioja': 'La Rioja',
    'Mendoza': 'Mendoza',
    'Misiones': 'Posadas',
    'Neuquén': 'Neuquén',
    'Río Negro': 'Viedma',
    'Salta': 'Salta',
    'San Juan': 'San Juan',
    'San Luis': 'San Luis',
    'Santa Cruz': 'Río Gallegos',
    'Santa Fe': 'Santa Fe',
    'Santiago del Estero': 'Santiago del Estero',
    'Tierra del Fuego': 'Ushuaia',
    'Tucumán': 'San Miguel de Tucumán',
    'Ciudad Autónoma de Buenos Aires': 'CIUDAD AUTONOMA DE BUENOS AIRES'}
    
    if row['CIUDAD'] == 'REMPLAZAR POR CAPITAL':
        return Provincias_capitales.get(row['PROVINCIA'], row['CIUDAD'])
    else:
        return row['CIUDAD']
    pass

def reemplazar_ciudad(row):
    if 'Ciudad Autónoma de Buenos Aires' in row['PROVINCIA']:
        return 'CIUDAD AUTONOMA DE BUENOS AIRES'
    else:
        return row['CIUDAD']
    
    pass

def texto_limpio(texto):
    texto_limpio = re.sub(r'\([^)]*\)', '', texto)
    return texto_limpio

    pass

def eliminar_caracteres_especiales(texto):
    # Utilizar una expresión regular para eliminar caracteres especiales excepto espacios
    texto_limpio = re.sub(r'[^\w\s()&?#/+\'`¡:¿°]', '', texto)
    return texto_limpio

    pass
