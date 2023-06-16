import numpy as np
from tqdm import tqdm
from scipy import optimize
from sklearn import preprocessing
import warnings
import matplotlib.pyplot as plt
import re


class Dataset:
    """
    Class of common dataarray (dim=2) methods. Data should be flattened beforehand otherwise.

    Public functions :
        Utils :
            # Dataset information
            returnMetadata(printMetadata=True) -> (categories, colomn_names, dimensions)
            returnData() -> (data, labels, theta)

            # Dataset visualization
            visualizeData2D(xaxis_col=0, yaxis_col=1, plot_type="scatter","default") -> fig
            visualizeData3D(xaxis_col=0, yaxis_col=1, zaxis_col=2, plot_type="scatter", "default") -> fig

        Data editing functions :
            # Dataset modification
            replaceData(previous_value, new_value) 

            # Dataset normalization
            normalizeData() -> X_normalized, X_mean, X_sigma
            denormalizeData() -> X_denormalized, X_mean, X_sigma
            
            # Data Principal Component Analysis (PCA)
            PCA(k_main_components=0, normalization=True) -> PCA array
            recoverDataPCA()
            visualizePCA()

        Data processing algorithms :
            # Gradient descent algorithms computation
            descentGradient(alpha=0.01, num_iters=50) -> theta
            gaussianGradient(alpha=0.01, num_iters=50) -> theta
            normalGradient(alpha=0.01, num_iters=50) -> theta

            # Gradient desecent algorithms evaluation
            predictGradient(test_data, test_label=0) -> prediction
            autoTestGradient() -> prediction, sucess_rate
            visualizeGradient() -> prediction, fig

            # OneVsAll classification
            oneVsAll(self, lambda_=0.1, tol=1e-8, itterations=50) -> theta
            predictOneVsAll(self, test_data, test_labels=0) -> prediction 

    """

    def __init__(self, data, labels, colomn_names,
                 theta=[0,0], alpha=1e4, num_iters=1000):
        
        # Mandatory data
        self.labels = labels
        self.data = data
        self.colomn_names = colomn_names

        # Additional intern variable
        self.normalizeDataMean = 0
        self.normalizeDataSigma = 1

        # Optional methods arguments
        self.theta = theta
        self.alpha = alpha
        self.num_iters = num_iters

    # Internal (private) functions ===========================================
    def __gaussianKernel(x1, x2, sigma):
        return np.exp(-np.linalg.norm(x1-x2)**2 / (2*sigma**2))

    def __sigmoid(z):
        g = 1/(1+np.exp(-z))
        return g
    
    def __computeCost(X, y, theta):

        m = y.shape[0]
        J = 0
        for i in range(m):
            h = np.dot(X[i, :], theta)
            J = J + 1/(2*m) * (h - y[i])**2

        return J

    def __lrCostFunction(theta, X, y, lambda_):

        m = y.size  
        J = 0
        grad = np.zeros(theta.shape)

        h = Dataset.__sigmoid(X@theta)
        J = (-y.T@np.log(h) - (1 - y).T@np.log(1 - h))/m

        grad = X.T @ (h - y)/m
        reg_term = lambda_ /(2*m) * (theta[1:].T@theta[1:])
        J = J + reg_term

        grad[1:] = grad[1:] + lambda_/m * theta[1:]
        
        return J, grad

    
    # Callable (public) functions ============================================

    # Utils ==================================================================
    def returnMetadata(self, printMetadata=True) -> tuple:
        categories = np.unique(self.labels)
        colomn_names = self.colomn_names
        dimensions = np.shape(self.data)

        if printMetadata:
            print("Categories : \n", categories)
            print("Colomns names : \n", colomn_names)
            print("Dimensions : \n", dimensions)
        return (categories, colomn_names, dimensions)
    
    def returnData(self) -> tuple:
        return (self.data, self.labels, self.theta)
    
    def visualizeData2D(self, xaxis_col=0, yaxis_col=1, plot_type="scatter"):
        xaxis_index = 0
        yaxis_index = 1
        for index, colomn_name in enumerate(self.colomn_names):
            if colomn_name == xaxis_col:
                xaxis_index = index
            elif colomn_name == yaxis_col:
                yaxis_index = index

        xaxis_data = self.data[:, xaxis_index]
        yaxis_data = self.data[:, yaxis_index]

        if plot_type == "default":
            fig = plt.figure()
            plt.plot(xaxis_data, yaxis_data)
        elif plot_type == "scatter":
            fig = plt.figure()
            plt.scatter(xaxis_data, yaxis_data)
        else:
            raise ValueError("Not a valid plot type")  
        
        plt.xlabel(xaxis_col)
        plt.ylabel(yaxis_col)
        plt.title("Colomns " + str(xaxis_col) + " VS " + str(yaxis_col))
        plt.grid(True, which="both")

        return fig

    def visualizeData3D(self, xaxis_col=0, yaxis_col=1, zaxis_col=2, plot_type="scatter"):
        xaxis_index = 0
        yaxis_index = 1
        zaxis_index = 2
        for index, colomn_name in enumerate(self.colomn_names):
            if colomn_name == xaxis_col:
                xaxis_index = index
            elif colomn_name == yaxis_col:
                yaxis_index = index
            elif colomn_name == zaxis_col:
                zaxis_index = index

        xaxis_data = self.data[:, xaxis_index]
        yaxis_data = self.data[:, yaxis_index]
        zaxis_data = self.data[:, zaxis_index]

        if plot_type == "default":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(xaxis_data, yaxis_data, zaxis_data)
        elif plot_type == "scatter":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xaxis_data, yaxis_data, zaxis_data)
        else:
            raise ValueError("Not a valid plot type")  

        ax.set_xlabel(xaxis_col)
        ax.set_ylabel(yaxis_col)
        ax.set_zlabel(zaxis_col)
        plt.title("Colomns " + str(xaxis_col) + " VS " + str(yaxis_col) + " VS " + str(zaxis_col))
        plt.grid(True, which="both")

        return fig
    
    # Data editing functions =================================================
    def replaceData(self, previous_value, new_value):
        """
        Replaces explicit data values by new ones, over the whole array
        """
        X = self.data
        
        X[X == previous_value] = new_value
        self.data = X
        return X

    def normalizeData(self):
        X = self.data
        X_norm = self.data

        X_mean = np.zeros(X_norm.shape[1])
        X_sigma = np.zeros(X_norm.shape[1])

        m, n = np.shape(X_norm)
        for j in range(n):
            X_mean[j] = np.mean(X_norm[:, j])
            X_sigma[j] = np.std(X_norm[:, j])
            for i in range(m):
                if X_sigma[j] != 0:
                    X_norm[i, j] = (X[i, j] - X_mean[j])/X_sigma[j]
                else : 
                    X_norm[i, j] = 0   
        
        # Storing normalization info, so the method can be reversed
        self.normalizeDataMean = X_mean
        self.normalizeDataSigma = X_sigma

        return X_norm, X_mean, X_sigma

    def denormalizeData(self):
        """
        (value - mean) / std
        """
        X = self.data
        X_norm = self.data

        X_mean = self.normalizeDataMean
        X_sigma = self.normalizeDataSigma
        print(X_mean, X_sigma)

        m, n = np.shape(X_norm)
        for j in range(n):
            for i in range(m):
                if X_sigma[j] != 0:
                    X_norm[i, j] = X[i, j]*X_sigma[j] + X_mean[j]
                else : 
                    X_norm[i, j] = 0   
        
        # Reseting normalization info
        self.normalizeDataMean = 0
        self.normalizeDataSigma = 1

        return X_norm, X_mean, X_sigma
    
    
    def PCA(self, k_main_components=0, normalization=True):
        X = self.data

        m, n = X.shape            
        if not k_main_components.any():
            k_main_components = n // 10

        cov = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort the eigenvectors by descending eigenvalues
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        
        # Compute the projection matrix
        X_proj_matrix = eigvecs[:, :k_main_components]

        # Project the image onto the reduced dimensional space
        X_pca = np.dot(X, X_proj_matrix)
        
        self.data_pca = X_pca
        self.data_pca_proj_matrix = X_proj_matrix

        return X_pca, X_proj_matrix


    def recoverDataPCA(self):

        try:
            X_pca, X_proj_matrix = self.data_pca, self.data_pca_proj_matrix
            X_reconstructed = np.dot(X_pca, X_proj_matrix)
        except:
            raise ValueError("Compute PCA first")
        
        self.data_recovered = X_reconstructed

        return X_reconstructed
    
    def visualizePCA(self):
        
        plots = [self.data, self.data_pca, self.data_recovered]
        titles = ["Data", "Data PCA", "Data recovered"]
        plt.subplots(1, 4)
        for i, data in enumerate(plots):
            plt.subplot(1, i)
            plt.imread(data)
            plt.title(titles[i])

        return None


    # Data processing functions ==============================================
    def descentGradient(self, alpha=0.01, num_iters=50):
        """
        Classic gradient descent algorithm 
        """
        X, y = self.data, self.labels

        m = y.shape[0] 
        J_history = []
        theta = np.zeros(X.shape[1])
        for k in tqdm(range(num_iters)):
            for i in range(m):
                h = np.dot(X[i, :], theta)
                theta = theta - alpha * 1/m * (X[i, :].T.dot((h-y[i])))   

        self.theta = theta

        return theta 

    def gaussianGradient(self, alpha=1e-4, num_iters=1000, sigma=1):
        """
        Gradient descent algorithm with gaussian weight
        """
        X, y = self.data, self.labels

        theta = np.zeros(X.shape[1])
        m = len(y)
        for k in tqdm(range(num_iters)):
            for i in range(m):
                h_i = np.dot(X[i], theta)
                grad = 0
                for j in range(m):
                    weight = Dataset.__gaussianKernel(X[i], X[j], sigma)
                    grad += weight * (h_i - y[i]) * X[i]
                    
                dJ = 1 / m * grad
                theta = theta - alpha * dJ
        self.theta = theta
        return theta  

    def normalGradient(self):
        """
        Exact gradient computation, heavy computation cost
        """
        X, y = self.data, self.labels

        theta = np.zeros(X.shape[1])
        theta = np.linalg.inv(X.T@X)@(X.T)@y
        self.theta = theta

        return theta
    
    def autoTestGradient(self):
        X, y = self.data, self.labels

        theta = self.theta
        length_test = len(y)

        prediction = np.round(X@theta)
        success_rate = np.sum(prediction == y) / length_test
        print("Gradient self-efficency =", round(success_rate, 4)*100, "%")

        return prediction
    
    def predictGradient(self, test_data, test_label=0):
        """
        test_data : input data to be classified using the previously computed gradient
        test_label : input labels to evaluate the gradient efficiency (optionnal)
        """

        theta = self.theta
        try:
            prediction = np.round(test_data@theta)
        except:
            raise ValueError("Gradient not compatible with this dataset")

        if test_label:
            success_rate = np.sum(prediction == test_label) / len(test_label)
            print(round(success_rate*100, 4))

        return prediction, success_rate
    
    def visualizeGradient2D(self, test_data, xaxis_col=0, yaxis_col=1):
        """
        Plots in 2D the data, labelized with their predicted categories
        """

        categories = np.unique(self.labels)
        print(categories)
        theta = self.theta
        try:
            prediction = np.round(test_data@theta)
        except:
            raise ValueError("Computed gradient not compatible with this dataset")
        
        xaxis_index = 0
        yaxis_index = 1
        for index, colomn_name in enumerate(self.colomn_names):
            if colomn_name == xaxis_col:
                xaxis_index = index
            elif colomn_name == yaxis_col:
                yaxis_index = index

        fig = plt.figure()
        for label in categories:
            plt.scatter(test_data[np.where(prediction==label), xaxis_index], 
                        test_data[np.where(prediction==label), yaxis_index], label=int(label))
        plt.legend(title="Categories")
        plt.xlabel(self.colomn_names[xaxis_index])
        plt.ylabel(self.colomn_names[yaxis_index])
        plt.title("Gradient classification prediction")

        return prediction, fig


    def visualizeGradient3D(self, test_data, xaxis_col=0, yaxis_col=1, zaxis_col=2):
        """
        Plots in 3D the data, labelized with their predicted categories
        """

        categories = np.unique(self.labels)
        print(categories)
        theta = self.theta
        try:
            prediction = np.round(test_data@theta)
        except:
            raise ValueError("Computed gradient not compatible with this dataset")
        
        xaxis_index = 0
        yaxis_index = 1
        zaxis_index = 2
        for index, colomn_name in enumerate(self.colomn_names):
            if colomn_name == xaxis_col:
                xaxis_index = index
            elif colomn_name == yaxis_col:
                yaxis_index = index
            elif colomn_name == zaxis_col:
                zaxis_index == index

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for label in categories:
            ax.scatter(test_data[np.where(prediction==label), xaxis_index], 
                        test_data[np.where(prediction==label), yaxis_index], 
                        test_data[np.where(prediction==label), zaxis_index], label=int(label))
        plt.legend(title="Categories")
        plt.xlabel(self.colomn_names[xaxis_index])
        plt.ylabel(self.colomn_names[yaxis_index])
        plt.title("Gradient classification prediction")

        return prediction, fig
    

    def oneVsAll(self, lambda_=0.1, tol=1e-8, itterations=50):
        """
        Classifies data by comparing the distance of a value to each category,
        and minimizing it.
        """    
        X, y = self.data, self.labels
    
        m, n = X.shape
        num_labels = len(np.unique(y))

        X = np.concatenate([np.ones((m, 1)), X], axis=1)
        options= {'maxiter': itterations}

        all_theta = np.zeros((num_labels, n + 1))
        Cost = np.zeros(num_labels)
        initial_theta = np.zeros(X.shape[1])
        for k in tqdm(range(num_labels)):
            y_k = np.zeros(m)
            y_k[y==k] = 1
            res = optimize.minimize(Dataset.__lrCostFunction, initial_theta, (X, y_k, lambda_ ), jac=True, method='SLSQP', tol=tol, options=options)
            Cost[k] = res.fun
            all_theta[k, :] = res.x
        
        self.theta = all_theta
        return all_theta

    def predictOneVsAll(self, test_data, test_labels=0):
        """
        test_data : input data to be classified using the previously computed gradient
        test_label : input labels to evaluate the gradient efficiency (optionnal)
        """

        all_theta = self.theta
        m, n = test_data.shape
        X = np.concatenate((np.ones((m, 1)), test_data), axis=1)

        prediction = np.zeros(X.shape[0])
        for i in range(m):
            prediction[i] = np.argmax(all_theta@X[i,:].T)

        if test_labels.any():
            sucess_rate = np.sum(prediction == test_labels) / len(prediction)
            print("OneVsAll success rate =", round(sucess_rate, 4)*100, "%")

        return prediction

if __name__ == "__main__":

    #================NEURAL NETWORK================# =========================

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

    def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_=0):

        Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                            (hidden_layer_size, (input_layer_size + 1)))

        Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                            (num_labels, (hidden_layer_size + 1)))

        m = y.size
        J = 0
        
        Theta1_grad = np.zeros(Theta1.shape)
        Theta2_grad = np.zeros(Theta2.shape)
        
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

    #================SVM================# ====================================

    from tqdm import tqdm

    def gaussianKernel(x1, x2, sigma):
        return np.exp(-np.linalg.norm(x1-x2)**2 / (2*sigma**2))

    def SVMtraining(X, y, X_test, y_test):

        C = 0.01
        sigma = 0.03
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
        
