def ova_gradient_descent(X, Y, eta, epochs, percent):
    '''Esta funcion se utiliza para implimentar el método de regresión lineal Batch Gradiente Descent
    batch_gradient_descent(X, Y, eta, epocs) where:
    X: DataFrame de instancias o features
    Y: DataFrame de targets
    eta: tasa de aprendizaje (learning rate)
    epochs: numero máximo de iteraciones
    percent: % de datos que seran utilizados para el test (base 100)
    
    ------------------------------------
    Return:
    In order: theta, test_index, train_index, Y_predict, J_log
    
    theta: valores correspondientes a theta_n
    test_index: data test index
    train_index: data training index
    Y_predict: Y predict values
    J_log: errores por numero de epoca
    Prob: Probabilidades de los 3 modelos
    Y_test: 
    
    '''
    import numpy as np
    import pandas as pd
    import random
    from sklearn.metrics import confusion_matrix
    
    m = len(X)
    test_index = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    train_index = list(np.arange(0, m))
    
    for element in test_index:
        train_index.remove(element)
        
    
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = np.c_[Y.iloc[train_index]]
    Y_test = np.c_[Y.iloc[test_index]]
    
    # Entrenamiento
    theta = np.array([np.zeros(5).reshape(-1,1), np.zeros(5).reshape(-1,1), np.zeros(5).reshape(-1,1)])
    yc = np.array([np.zeros(len(Y_train)).reshape(-1,1), np.zeros(len(Y_train)).reshape(-1,1), np.zeros(len(Y_train)).reshape(-1,1)])
   
    for i in range(3):
        yc[i] = (Y_train == i).astype(np.int32).reshape(-1,1)
        theta[i] = np.random.randn((X.shape[1] + 1), 1)    

    m = len(X_train)
    
    X_b = np.c_[np.ones((m, 1)), X_train]
        
    for i in range(3):
        J_log = np.zeros(epochs)
        theta_t = theta[i]
        y_train_t = yc[i]
        for j in range(epochs):
            J_log[j] =(-1/m)*(y_train_t*np.log(sigmoid(X_b @ theta_t)) + (1-y_train_t)*(np.log(1-sigmoid(X_b @ theta_t)))).sum(axis=0)
            gradients = (1 / m) * (X_b.T @ (sigmoid(X_b @ theta_t) - y_train_t))
            theta_t = theta_t - eta * gradients                       
        theta[i] = theta_t                               

    # Test
    
    m = len(X_test)
    
    X_b_test = np.c_[np.ones((m, 1)), X_test]
    
    y_pr0 = np.zeros(len(test_index))
    y_pr1 = np.zeros(len(test_index))
    y_pr2 = np.zeros(len(test_index))
   
    y_pr= np.array([y_pr0, y_pr1, y_pr2])  
    
    for i in range(3):
        y_pr[i] = sigmoid(theta[i].T @ X_b_test.T)
        
    Prob = np.c_[y_pr[0:1].T , y_pr[1:2].T , y_pr[2:3].T ]    
    Y_predict = np.argmax(Prob, axis=-1)    

    return theta, test_index, train_index, Y_predict, J_log, Prob

def sigmoid(z):
    import numpy as np
    return 1/(1+np.exp(-z))

def ovo_gradient_descent(X, Y, eta, epochs, percent):
    '''Esta funcion se utiliza para implimentar el método de regresión lineal Batch Gradiente Descent
    batch_gradient_descent(X, Y, eta, epocs) where:
    X: DataFrame de instancias o features
    Y: DataFrame de targets
    eta: tasa de aprendizaje (learning rate)
    epochs: numero máximo de iteraciones
    percent: % de datos que seran utilizados para el test (base 100)
    
    ------------------------------------
    Return:
    In order: theta, test_index, train_index, Y_predict, J_log
    
    theta: valores correspondientes a theta_n
    test_index: data test index
    train_index: data training index
    Y_predict: Y predict values
    J_log: errores por numero de epoca
    Prob: Probabilidades de los 3 modelos
    Y_test: 
    
    '''
    import numpy as np
    import pandas as pd
    import random
    from sklearn.metrics import confusion_matrix
    
    m = len(X)
    test_index = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    train_index = list(np.arange(0, m))
    
    for element in test_index:
        train_index.remove(element)
        
    
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = np.c_[Y.iloc[train_index]]
    Y_test = np.c_[Y.iloc[test_index]]
    
    A = pd.concat([X.iloc[train_index], Y.iloc[train_index]] , axis = 1)
    #A = np.c_[X_train, Y_train]
    
    y0 = (Y_train==0).astype(np.int32).reshape(-1,1)
    y1 = (Y_train==1).astype(np.int32).reshape(-1,1)
    y2 = (Y_train==2).astype(np.int32).reshape(-1,1)
    
    #b = (Y_train == 0)
    
    Xtrain_zero = A[A.iloc[:,4] == 0 ].iloc[:,[0,1,2,3]]      # X_train de Clase 0 
    Xtrain_one = A[A.iloc[:,4] == 1 ].iloc[:,[0,1,2,3]]       # X_train de Clase 1 
    Xtrain_two = A[A.iloc[:,4] == 2 ].iloc[:,[0,1,2,3]]       # X_train de Clase 2
    
    Xtrain01 = np.append(Xtrain_zero, Xtrain_one, axis=0)
    Xtrain02 = np.append(Xtrain_zero, Xtrain_two, axis=0)
    Xtrain12 = np.append(Xtrain_one, Xtrain_two, axis=0)
    
    Xtrain = np.array([Xtrain01, Xtrain01, Xtrain12])
    
    # Entrenamiento
    theta = np.array([np.zeros(5).reshape(-1,1), np.zeros(5).reshape(-1,1), np.zeros(5).reshape(-1,1)])
    #yc = np.array([np.zeros(len(Y_train)).reshape(-1,1), np.zeros(len(Y_train)).reshape(-1,1), np.zeros(len(Y_train)).reshape(-1,1)])
   
    #np.sum
    y01 = np.append(np.zeros(np.sum(y0)), np.ones(np.sum(y1)))       #Clase 0 vs 1
    y02 = np.append(np.zeros(np.sum(y0)), 2*(np.ones(np.sum(y2))))   #Clase 0 vs 2
    y12 = np.append(np.ones(np.sum(y1)), 2*(np.ones(np.sum(y2))))    #Clase 1 vs 2
    
    yc = np.array([y01,y02,y12])
    
    for i in range(3):
        theta[i] = np.random.randn((X.shape[1] + 1), 1)    

    m = len(X_train)
    m01 = len(Xtrain01)
    m02 = len(Xtrain02)
    m12 = len(Xtrain12)
   
    
    #X_b = np.c_[np.ones((m, 1)), X_train]
    #X_b = np.c_[np.ones((m, 1)), Xtrain]
    
    X_b01 = np.c_[np.ones((m01, 1)), Xtrain01]              #Clase 0 vs 1
    X_b02 = np.c_[np.ones((m02, 1)), Xtrain02]               #Clase 0 vs 2
    X_b03 = np.c_[np.ones((m12, 1)), Xtrain12]                #Clase 1 vs 2
             
    #for i in range(3):
        #J_log = np.zeros(epochs)
        #theta_t = theta[i]
        #y_train_t = yc[i]
        #for j in range(epochs):
            #J_log[j] =(-1/m)*(y_train_t*np.log(sigmoid(X_b @ theta_t)) + (1-y_train_t)*(np.log(1-sigmoid(X_b @ theta_t)))).sum(axis=0)
            #gradients = (1 / m) * (X_b.T @ (sigmoid(X_b @ theta_t) - y_train_t))
            #theta_t = theta_t - eta * gradients                       
        #theta[i] = theta_t                               
     
   
    theta1 = theta[1]
    y_train1 = yc[1]
    J_log = np.zeros(epochs)
    for j in range(epochs):
        J_log =(-1/m)*(y_train1*np.log(sigmoid(X_b01 @ theta1)) + (1-y_train1)*(np.log(1-sigmoid(X_b01 @ theta1)))).sum(axis=0)
        gradients = (1 / m) * (X_b01.T @ (sigmoid(X_b01 @ theta1) - y_train1))
        theta1 = theta1- eta * gradients                       
    theta[1] = theta1
    
    theta2 = theta[2]
    y_train2 = yc[2]
    J_log = np.zeros(epochs)
    for j in range(epochs):
        J_log =(-1/m)*(y_train2*np.log(sigmoid(X_b02 @ theta2)) + (1-y_train2)*(np.log(1-sigmoid(X_b02 @ theta2)))).sum(axis=0)
        gradients = (1 / m) * (X_b02.T @ (sigmoid(X_b02 @ theta2) - y_train2))
        theta2 = theta2- eta * gradients                       
    theta[2] = theta2
    
    theta3 = theta[3]
    y_train3 = yc[3]
    J_log = np.zeros(epochs)
    for j in range(epochs):
        J_log =(-1/m)*(y_train3*np.log(sigmoid(X_b03 @ theta3)) + (1-y_train3)*(np.log(1-sigmoid(X_b03 @ theta3)))).sum(axis=0)
        gradients = (1 / m) * (X_b03.T @ (sigmoid(X_b03 @ theta3) - y_train3))
        theta3 = theta3- eta * gradients                       
    theta[3] = theta3
    # Test
    
    m = len(X_test)
    
    X_b_test = np.c_[np.ones((m, 1)), X_test]
    
    y_pr0 = np.zeros(len(test_index))
    y_pr1 = np.zeros(len(test_index))
    y_pr2 = np.zeros(len(test_index))
   
    y_pr= np.array([y_pr0, y_pr1, y_pr2])  
    
    for i in range(3):
        y_pr[i] = sigmoid(theta[i].T @ X_b_test.T)
        
    Prob = np.c_[y_pr[0:1].T , y_pr[1:2].T , y_pr[2:3].T ]    
    Y_predict = np.argmax(Prob, axis=-1)    

    return theta, test_index, train_index, Y_predict, J_log, Prob, Xtrain

















