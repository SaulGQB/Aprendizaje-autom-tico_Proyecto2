def plot_image(i, proba_array, true_label, img):
    """
    i: indice de la matriz de datos
    proba_array: arreglo con las probabilidades de predicción de las clases para cada imágen
    true_label: matriz con los valores reales de las clases para cada imágen
    img: matriz de imágenes (features)
    """
    import numpy as np
    import matplotlib.pyplot as plt 
    
    proba_array, true_label, img = proba_array, true_label[i], img[i] # Con i se extrae fila por fila de datos
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.reshape(28,28), cmap=plt.cm.binary)

    predicted_label = np.argmax(proba_array) # Se extrae el indice donde la probabilidad es mayor, este coincide con la clase
    if predicted_label == true_label:
        color = 'blue' # Para el nombre, azul si se determinó correctamente la clase
    else:
        color = 'red' # Rojo si no se determinó correctamente la clase

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], # Extrae el nombre de la clase de acuerdo a lo predicho
                                100*np.max(proba_array), # Extrae la probabilidad mayor y la convierte en porcentaje
                                class_names[true_label]), # Extrae el nombre de la clase de la matriz class_names
                                color=color) # Se define el color de la impresión

def plot_value_array(i, proba_array, true_label):
    """
    i: indice de la matriz de datos (para usar fácilmente con ciclo for)
    predictios_array: arreglo con las probabilidades de predicción de las clases para cada imágen
    true_label: matriz con los valores reales de las clases para cada imágen
    """
    import numpy as np
    import matplotlib.pyplot as plt 
    
    proba_array, true_label = proba_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10)) # Se indica que habrá espacio para las 10 clases, se comienza en 0 y termoina en 9
    plt.yticks([])
    thisplot = plt.bar(range(10), proba_array, color="#777777") # Se establece el graficado de las probabilidades de la predicción para la imágen
    plt.ylim([0, 1]) # al no ser mayor a 1 se delimita el eje y en 1 como máximo (1 == 100% de coincidencia con la clase)
    predicted_label = np.argmax(proba_array) # Se extrae la clase con mayor probabilidad (numero entero)

    thisplot[predicted_label].set_color('red') # El predicho lo configura como rojo
    thisplot[true_label].set_color('blue') # El real lo configura como azul