import numpy as np
import math
from scipy import optimize

class NeuralNetworkRegressor:
	"""Neural network used for regression, with a single hidden layer"""

	def __init__(self,n_features,n_hiddenlayer,n_examples):
		"""Initialize the NN class"""

		#number of features and training data
		self.n_features = n_features
		self.n_hiddenlayer = n_hiddenlayer
		self.n_examples = n_examples

		#initialize the weights
		theta_range = 1.0
		self.Theta1 = theta_range*(2*np.random.rand(n_hiddenlayer,n_features)-1)
		self.Theta2 = theta_range*(2*np.random.rand(n_hiddenlayer+1)-1)


	def predict(self,X):
		"""Return the NN's prediction given a set of feature vectors X.
		Note x includes an extra 1 as its first element"""

		z = np.dot(X,np.transpose(self.Theta1))
		a = sigmoid(z)
		a = np.column_stack((np.ones(X.shape[0]),a)) #bias unit
		z2 = np.dot(a,np.transpose(self.Theta2))
		return sigmoid(z2)

	def costFunction(self,params,X,y,alpha=0.0):
		"""Cost function of training data (X,y)
		Returns a tuple: the value of the cost function and its gradient:
		(cost, grad wrt theta1, grad wrt theta 2). 
		'params' are the Theta matrices, flattened"""

		m=float(self.n_examples)

		#reshape params into Theta arrays
		Theta1,Theta2 = np.split(params,[self.n_features*self.n_hiddenlayer])
		Theta1 = Theta1.reshape((self.n_hiddenlayer,self.n_features))

		#make predictions on X in a vectorized way
		z = np.dot(X,np.transpose(Theta1))
		a = sigmoid(z)
		a = np.column_stack((np.ones(self.n_examples),a)) #bias unit
		z2 = np.dot(a,np.transpose(Theta2))
		pred = sigmoid(z2)

		#go through examples one by one
		#and add up cost fn

		summand = 0
		for idx,example in enumerate(X):
			
			try:
				summand = summand  - y[idx]*math.log(pred[idx]) - (1-y[idx])*math.log(1-pred[idx])
			except:
				print "Error calculating cost fn.", y[idx],pred[idx],z2[idx]

		regularization = alpha * ( sum(sum(Theta1[:,1:]*Theta1[:,1:])) + sum(Theta2[1:]*Theta2[1:]) )

		cost = (summand + 0.5 * regularization)/m

		#calcualte gradients of the cost function using backpropagation algorithm
		Theta1_grad = np.zeros(Theta1.shape)
		Theta2_grad = np.zeros(Theta2.shape)

		for idx,example in enumerate(X):

			delta3 = pred[idx] - y[idx]
			delta2 = (delta3*Theta2)[1:]*sigmoidGradient(z[idx]) 

			Theta1_grad = Theta1_grad + np.outer(delta2,example)
			Theta2_grad = Theta2_grad + delta3*a[idx]

		Theta1_grad = Theta1_grad/m
		Theta2_grad = Theta2_grad/m

		Theta1_grad[:,1:] = Theta1_grad[:,1:] + (alpha/m)*Theta1[:,1:]
		Theta2_grad[1:] = Theta2_grad[1:] + (alpha/m)*Theta2[1:]

		grad_flat = np.concatenate((np.ravel(Theta1_grad),Theta2_grad),axis=1)

		return cost, grad_flat

	def costFunctionNumGrad(self,params,X,y):
		"""Calculate numerical gradient of the cost fn"""

		numgrad = np.zeros(len(params))
		eps = 0.00001

		#cost=self.costFunction(params,X,y)[0]

		for i in range(len(params)):
			params_plus = params + eps*np.equal( range(len(params)), i)
			params_minus = params - eps*np.equal( range(len(params)), i)
			numgrad[i] = self.costFunction(params_plus,X,y)[0] - self.costFunction(params_minus,X,y)[0]
			numgrad[i] = numgrad[i]/(2*eps)


		return numgrad


	def train(self,X,y):
		"""Train the NN with training examples, where
		X is a matrix containing the features for each example and
		y a vector with the targets; y must be normalized so all targets
		are in (0,1)"""

		#flatten out and concatenate the Thetas
		params = np.concatenate((np.ravel(self.Theta1),self.Theta2),axis=1)

		#minimize the cost finction over the Theta arrays
		res=optimize.minimize(self.costFunction,params,args=(X,y),jac=True,method='CG')
		params = res.x

		#reshape 'params' back into the original form
		Theta1,Theta2 = np.split(params,[self.n_features*self.n_hiddenlayer])

		self.Theta1 = Theta1.reshape((self.n_hiddenlayer,self.n_features))
		self.Theta2 = Theta2

		return self.costFunction(params,X,y)[0]


	def gradientTest(self,X,y):
		"""Check the backprop algorithm by checking against numerical gradient of costFunction"""

		params = np.concatenate((np.ravel(self.Theta1),self.Theta2),axis=1)

		ng = self.costFunctionNumGrad(params,X,y)
		cf =  self.costFunction(params,X,y)

		print "costFunction: ", cf[0]
		print "grad (backprop): ", cf[1]
		print "grad (num): ", ng

	def printThetas(self):

		print "Theta1 :", self.Theta1
		print "Theta2: ", self.Theta2


def sigmoid(x):
	try:
		return 1/(1+math.exp(-x))
	except:
		print "Sigmoid error: x=", x 
		return 0


def sigmoidGradient(x):
	return sigmoid(x)*(1-sigmoid(x))

#vectorize sigmoid functions
sigmoid = np.vectorize(sigmoid)
sigmoidGradient = np.vectorize(sigmoidGradient)



