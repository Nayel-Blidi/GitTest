import numpy as np
from tqdm import tqdm
from scipy import optimize
from sklearn import preprocessing
from scipy.io import loadmat
import utils

#%% 
#================GRADIENT REGRESSION================#

#%%
#Stocastic gradient descend 2 colomns (2 variables)

def gaussianKernel(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1-x2)**2 / (2*sigma**2))


def computeCost(X, y, theta):
    #COMPUTECOST Compute cost for linear regression
    #   using theta as the parameter for linear regression to fit the data points in X and y
    
    # Initialize some useful values
    m = y.shape[0] # number of training examples

    #You need to return the following variables correctly 
    J = 0;

    # ============================================================
    for i in range(m):
        h = theta[0] + theta[1]*X[i]
        J = J + 1/(2*m) * (h - y[i])**2
    
    # ============================================================
    return J

def gaussianGradient(X, y, theta, alpha, num_iters, sigma=1):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    
    for k in tqdm(range(num_iters)):
        for i in range(m):
            h_i = np.dot(X[i], theta)
            grad = 0
            for j in range(m):
                weight = gaussianKernel(X[i], X[j], sigma)
                grad += weight * (h_i - y[i]) * X[i]
                
            dJ = 1 / m * grad
            theta = theta - alpha * dJ
            
        J_history.append(computeCost(X, y, theta))
        
    return theta, J_history  

Yy = np.linspace(0, 10, 10)
Xx = np.random.rand(1, 10).T
Xx = np.concatenate([np.ones((10, 1)), Xx], axis=1)

theta = np.random.rand(2)
alpha = 1e-4
num_iters = 1000

#theta_, J_history = gaussianGradient(Xx, Yy, theta, alpha, num_iters)

#%% Normalization
#Normalized X and concatenation of a colomn of ones
    
def featureNormalize(X):
    """Normalized X and concatenation of a colomn of ones"""
    
    #FEATURENORMALIZE Normalizes the features in X 
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X_norm.shape[1])
    sigma = np.zeros(X_norm.shape[1])
    
    # ===========================================================
    m, n = np.shape(X_norm)
    for j in range(n):
        mu[j] = np.mean(X_norm[:, j])
        sigma[j] = np.std(X_norm[:, j])
        for i in range(m):
            X_norm[i, j] = (X[i, j] - mu[j])/sigma[j]
    

    # ============================================================
    
    
    return X_norm, mu, sigma

#%%
#Stocastic Gradient descend multiple colomns (variables)

def computeCostMulti(X, y, theta):

    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    # ===============================================================
    theta = theta.copy()

    for i in range(m):
        h = np.dot(X[i, :], theta)
        J = J + 1/(2*m) * (h - y[i])**2

    # ==================================================================
    return J


def gradientDescentMulti(X, y, theta, alpha=0.01, num_iters=50):
    """Stocastic Gradient descend multiple colomns (multi variables)"""
    
    #GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha
    
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    # =============================================================
    
    for k in tqdm(range(num_iters)):
        #dJ = np.zeros((m, 1))
        for i in range(m):
            #theta[0] can be directly computed thanks to the colomn of ones concatenated to X
            #It is thus the same algorithn as the gradientDescent one
            h = np.dot(X[i, :], theta)
            
            #dot computed the product of h-y times one row of cells of X[i]
            theta = theta - alpha * 1/m * (X[i, :].T.dot((h-y[i])))
        # Save the cost J in every iteration    
        J_history.append(computeCostMulti(X, y, theta))
        
    # ==============================================================     

    return theta, J_history 

Yy = np.linspace(0, 10, 10)
Xx = np.random.rand(1, 10).T
Xx = np.concatenate([np.ones((10, 1)), Xx], axis=1)

theta = np.random.rand(2)
alpha = 1e-4
num_iters = 10000

#theta_, J = gradientDescentMulti(Xx, Yy, theta, alpha, num_iters)

#%%
#Exact calculation of a gradient

def normalEqn(X, y):
    """Exact Gradient calculation multiple colomns (multi variables)"""

    theta = np.zeros(X.shape[1])
    
    # ================================================================
    theta = np.linalg.inv(X.T@X)@(X.T)@y
    
    # =================================================================
    return theta

#%% 
#================GRADIENT CLASSIFICATION================#

#%%
import numpy as np

#Définition de la fonction d'activation sigmoid h
def sigmoid(z):
    z = np.array(z)
    
    # You need to return the following variables correctly 
    g = np.zeros(z.shape)
    
    g = 1/(1+np.exp(-z))

    return g

#Fonction de coût régularisée (regularized logistic regression)
def lrCostFunction(theta, X, y, lambda_):
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)
    
    # ====================== YOUR CODE HERE ======================
    
    h = sigmoid(X@theta)
    
    J = (-y.T@np.log(h) - (1 - y).T@np.log(1 - h))/m
    grad = X.T @ (h - y)/m

    reg_term = lambda_ /(2*m) * (theta[1:].T@theta[1:])
    
    J = J + reg_term
    grad[1:] = grad[1:] + lambda_/m * theta[1:]
    
    # =============================================================
    
    return J, grad

#%%
from tqdm import tqdm
from scipy import optimize

#Calcul en entraînant 1 catégorie contre toutes les autres, pour établir le theta qui 
#générera un vecteur de probabilité d'appartenir à chacune des classes apprises
def oneVsAll(X, y, num_labels, lambda_, tol=1e-8, itterations=50):    
   

    # Some useful variables
    m, n = X.shape;
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1));
    Cost = np.zeros(num_labels);

    #X, mean, sigma = featureNormalize(X)
    
    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1);
    initial_theta = np.zeros(X.shape[1]);
    # set options for optimize.minimize
    options= {'maxiter': itterations}

    # ====================== YOUR CODE HERE ======================
    
    for k in tqdm(range(num_labels)):
        y_k = np.zeros(m)
        y_k[y==k] = 1
        res = optimize.minimize(lrCostFunction, initial_theta, (X, y_k, lambda_ ), jac=True, method='SLSQP', tol=tol, options=options)
        Cost[k] = res.fun
        all_theta[k, :] = res.x
    
    # =============================================================
    return Cost, all_theta

#%%
#Discrimination des résultats, catégorie = probabilité max d'appartenir aux classes listées 
def predictOneVsAll(all_theta, X):    
  
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m, n = X.shape;
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================
    # make a copy of X, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    X_bis = X.copy()

    for i in range(m):
        p[i] = np.argmax(all_theta@X_bis[i,:].T)

    # =============================================================
    return p

#%% 
#================NEURAL NETWORK================#

#%%
from tqdm import tqdm
from scipy import optimize
from scipy.io import loadmat
import utils

def sigmoidGradient(z):

    g = np.zeros(z.shape)

    g = 1/(1+np.exp(-z)) * (1-1/(1+np.exp(-z)))

    return g

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):

    W = np.zeros((L_out, 1 + L_in))
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    return W

#%%

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_=0):

    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    m = y.size
    J = 0
    
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # =============================================================
    
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    z2 = a1 @ Theta1.T
    a2 = np.concatenate([np.ones((m, 1)), 1 / (1 + np.exp(-z2))], axis=1)
    h = 1 / (1 + np.exp(-a2 @ Theta2.T))
    
    for k in range(num_labels):
        y_k = (y == k).astype(int)
        for i in range(m):
            J = J + 1/m * (-y_k[i] * np.log(h[i, k]) - (1 - y_k[i]) * np.log(1 - h[i, k]))
    
    regularization = 0
    for j in range(0, hidden_layer_size):
        for k in range(1, X.shape[1] + 1):
            regularization += Theta1[j, k]**2
    for j in range(0, num_labels):
        for k in range(1, hidden_layer_size + 1):
            regularization += Theta2[j, k]**2
    
    regularization = regularization * lambda_/(2*m)
    J = J + regularization
        
    error3 = h - (y[:, np.newaxis] == np.arange(num_labels))
    error2 = (error3@Theta2)[:, 1:] * sigmoidGradient(z2)

    Delta2 = error3.T @ a2
    Delta1 = error2.T @ a1
    
    Theta2_grad_regularization = lambda_ * np.concatenate([np.zeros((num_labels, 1)), Theta2[:, 1:]], axis=1)
    Theta1_grad_regularization = lambda_ * np.concatenate([np.zeros((hidden_layer_size, 1)), Theta1[:, 1:]], axis=1)
    
    Theta2_grad = 1/m * (Delta2 + Theta2_grad_regularization)
    Theta1_grad = 1/m * (Delta1 + Theta1_grad_regularization)
    
    # ================================================================
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
   
    return J, grad


def nnOneLayer(X, y, lambda_ = 0.1, hidden_layer_size = 32, num_labels = 2, itterations = 100):
    """
    Input shape : (m exemples , n data de l'exemple)
    y shape : m labels de catégories
    Output shape : (m labels de catégories , num_labels nombre de categories)
    """
    
    input_layer_size=np.shape(X)[1]
    
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    
    # Unroll parameters
    initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
    
    options= {'maxiter': itterations}
    
    costFunction = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)
    res = optimize.minimize(costFunction,initial_nn_params,jac=True,method='TNC',options=options)
    nn_params = res.x
    
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    
    try:
        pred = utils.predict(Theta1, Theta2, X)
        print('nnOneLayer Training Accuracy: %f' % (np.mean(pred == y) * 100))
    except:
        None
    
    return Theta1, Theta2


def predOneLayer(X, Theta1, Theta2):
    
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros(X.shape[0])

    #Network procesing and y (a3) calculation
    X = np.concatenate([np.ones([m, 1]), X], axis=1)    #Layer 0
    z2 = X@Theta1.T
    a2 = 1 / (1 + np.exp(-z2))                          #Layer 1 (neurons)

    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)
    z3 = a2 @ Theta2.T
    a3 = 1 / (1 + np.exp(-z3))                          #Layer 2 (output)

    for k in range(m):
        p[k] = np.argmax(a3[k])
    
    return p

def predOneLayerInv(y, Theta1, Theta2):

    m = y.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros(X.shape[0])

    #Network procesing and y (a3) calculation
    X = np.concatenate([np.ones([m, 1]), X], axis=1)    #Layer 0
    z2 = X@Theta1.T
    a2 = 1 / (1 + np.exp(-z2))                          #Layer 1 (neurons)

    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)
    z3 = a2 @ Theta2.T
    a3 = 1 / (1 + np.exp(-z3))                          #Layer 2 (output)
    
    
    
    return p


def utilspredOneLayer(X, Theta1, Theta2):
    return utils.predict(Theta1, Theta2, X)

#%% 
#================SVM================#

#%%
from tqdm import tqdm

def gaussianKernel(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1-x2)**2 / (2*sigma**2))

def SVMtraining(X, y, X_test, y_test):

   #You need to return the following variables correctly.
    C = 0.01;
    sigma = 0.03;
    # ====================== YOUR CODE HERE ======================
    valeurs = []
    for k in range(4):
        for j in range(4):
            valeurs.append((0.01*10**k, 0.01*10**j))
            valeurs.append((0.01*10**k, 0.03*10**j))
            valeurs.append((0.03*10**k, 0.01*10**j))
            valeurs.append((0.03*10**k, 0.03*10**j))
    
    results = []
    for k in tqdm(range(len(valeurs))):
            model = utils.svmTrain(X, y, valeurs[k][0], gaussianKernel, args=(valeurs[k][1],))
            predictions = utils.svmPredict(model, X_test)
            results.append(np.mean(predictions != y_test)*100)

    index = np.argmin(results)
    C, sigma = valeurs[index]
    print("Taux d'échec = ", min(results), " %\n", 
          "Valeurs (C, sigma) optimales = ", valeurs[index])

    return C, sigma

def SVM_plot(X, y, C, sigma):
    model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
    utils.visualizeBoundary(X, y, model)
    return None


#%%

def findClosestCentroids(X, centroids):
    # Set the number of clusters
    K = centroids.shape[0]

    # The centroids are in the following variable
    idx = np.zeros(X.shape[0], dtype=int)
    
    for i in range(X.shape[0]):
        distance = np.linalg.norm(X[i]-centroids,axis=1)
        idx[i] = np.argmin(distance)

    return idx

def computeCentroids(X, idx, K):
    m, n = X.shape
    # The centroids are in the following variable
    centroids = np.zeros((K, n))

    for k in range(K):
        X_k = X[idx == k]
        centroids[k] = np.mean(X_k, axis = 0)
    
    return centroids

def kMeansInitCentroids(X, K):

    m, n = X.shape
    
    # You should return this values correctly
    centroids = np.zeros((K, n))
    
    randidx = np.random.permutation(m)
    centroids = X[randidx[:K], :]  
    
    return centroids

#%%

def pca(X, k_main_components, normalization=False):

    #PCA Run principal component analysis on the dataset X
    #U, S = pca(X) computes eigenvectors of the covariance matrix of X
    #Returns the eigenvectors U, the eigenvalues (on diagonal) in S

    # Useful values
    m, n = X.shape
    
    if normalization:
        """
        scaler = preprocessing.StandardScaler().fit(X)
        X_norm, mean, std = scaler.transform(X), scaler.mean_, scaler.scale_
        """
        X_norm, mean, std = featureNormalize(X)
    else:
        X_norm, mean, std = X, 0, 1
        
    cov = np.cov(X_norm, rowvar=False)
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort the eigenvectors by descending eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    
    # Compute the projection matrix
    X_proj_matrix = eigvecs[:, :k_main_components]

    # Project the image onto the reduced dimensional space
    X_pca = np.dot(X_norm, X_proj_matrix)
    
    return X_pca, (X_proj_matrix, mean, std)


def pca_recoverData(X_pca, pca_info):
    #RECOVERDATA Recovers an approximation of the original data when using the 
    #projected matrix of the original data
        
    # Reconstruct the image using the reduced dimensional space
    X_reconstructed = (np.dot(X_pca, pca_info[0].T) + pca_info[1]) * pca_info[2]
    
    return X_reconstructed





    
    
