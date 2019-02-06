import numpy as np

class LinearRegressor:

    def __init__(self):
        self.theta = None


    def train(self,X,y,learning_rate=1e-3, num_iters=100,verbose=False):

        """
        Train a linear model using gradient descent.
        
        Inputs:
        - X: 1-dimensional array of length N of training data. 
        - y: 1-dimensional array of length N with values in the reals.
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
    
        J_history = []

        # Initialize self.theta
        if self.theta is None:
            # lazily initialize theta 
            self.theta = np.zeros((X.shape[1],))

        # Run gradient descent to find theta
        for i in range(num_iters):
            # evaluate loss and gradient
            loss, grad = self.loss(X, y)

            # add loss to J_history
            J_history.append(loss)

            #########################################################################
            # TODO:                                                                 #
            # Update the parameters using the gradient and the learning rate.       #
            #    One line of code expected                                          #
            #########################################################################
            self.theta = self.theta - (learning_rate*grad)


            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # print loss every 1000 iterations
            if verbose and i % 1000 == 0:
                print ('iteration %d / %d: loss %f' % (i, num_iters, loss))

        return J_history

    def loss(self, X, y):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - X: vector of length N with real values
        - y: 1-dimensional array of length N with real values.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        pass

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: vector of length N of training data. 

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is a real number.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted outputs in y_pred.           #
        #    One line of code expected                                            #
        ###########################################################################
        y_pred = np.dot(X,self.theta)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred
  

class LinearReg_SquaredLoss(LinearRegressor):
    "A subclass of Linear Regressors that uses the squared error loss function """

    """
    Function that returns loss and gradient of loss with respect to (X, y) and
    self.theta
        - loss J is a single float
        - gradient with respect to self.theta is an array of the same shape as theta

    """

    def loss (self,X,y):
        J = 0
        grad = np.zeros((2,))
        # print(self.theta)
        ###########################################################################
        # TODO:                                                                   #
        # Calculate J (loss) and grad (gradient) wrt to X,y, and self.theta.      #
        #   2-4 lines of code expected                                            #
        ###########################################################################
        m = len(X)
        h_x = np.dot(X,self.theta)
        # print(h_x.shape,y.shape)
        # J = (np.transpose(((np.dot(X,self.theta))-y)).dot((np.dot(X,self.theta))-y))/2*m
        J = np.sum((h_x-y)**2)/(2.0*m)
        # grad = ((np.transpose(X)).dot((np.dot(X,self.theta))-y))/m
        grad = np.dot((h_x-y),X)*(1.0/m)
        # grad = (np.sum((h_x-y).dot(X)))/m
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return J, grad
