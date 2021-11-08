def outliers_detect(df, column):
    '''Función para detectar outliers a partir de los cuantiles
    df: dataframe 
    column: columna del data frame
    -------------------------------------
    return: regresa una lista con los indices de los outliers
    '''
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 -1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers_index = df.index[(df[column] < lower) | (df[column] > upper)]
    
    return outliers_index

 
def remove_outliers(df, index_list):
    '''Funcion para remover los outliers de un dataframe
    df: data frame
    index_list: lista de indices donde se encuentran los outliers
    ------------------------------------------------
    return: Dataframe sin outliers
    '''
    index_list = sorted(set(index_list))
    df = df.drop(index_list)
    return df   

     
def outliers_replace(df, index_list, value):
    '''Reemplaza el valor de los outliers
    df: data frame
    index_list: lista de indices donde se encuentran los outliers
    value: valor con el que se reemplazaran 
    ------------------------------------------------
    return: Dataframe con los outliers reemplazados
    '''
    index_list = sorted(set(index_list))
    df.iloc[index_list] = value
    return df
