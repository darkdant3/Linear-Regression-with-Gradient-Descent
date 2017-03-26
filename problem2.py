import numpy as np
import sys
import matplotlib
#matplotlib.use('ggplot2')
import matplotlib.pyplot as plt
from copy import deepcopy
import os


class LinearRegression:
    ''' Linear regression algorithm implementation with gradient descent optimization  '''

    def __init__(self):
        self.iterations = 0
        self.max_iterations = 0
        self.cost_list = []
        self.beta_list = []
        self.alpha = 0.01
        self.decimals = 2


    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize betas
        self.beta = np.zeros(n_features)  # b0 = 0

        for iteration in np.arange(self.max_iterations):

            # calculate cost
            self.cost_list.append(deepcopy(self.cost_j(X, y)))
            # update  betas
            self.beta_list.append(deepcopy(self.gradient_descent(X, y)))

            self.iterations += 1

            # convergence test
            # compare the difference between the last two weight sets
            try:
                if np.round(self.cost_list[-1], self.decimals) == 0:
                    print 'Converged,  beta={}'.format(self.beta_list[-1])
                    print '%d Iterations' % self.iterations
                    print 'Cost J=%f Alpha=%f' % (self.cost_list[-1], self.alpha)
                    return
            except IndexError as e:
                pass

    def predict(self, x):
        # h(x) = beta.transpose() * x = sum(x[j]*beta[j])
        n_features = len(x)
        prediction = 0
        for j in np.arange(n_features):
            prediction += x[j] * self.beta[j]

        return prediction


    def cost_j(self, X, y):
        ''' Calculates Square error of predicted  value to the actual label
        J(beta_j) = 1/(2*m) * sum(h(x_i) - y_i ))**2 = 1/2m * transpose(X*betas - y) * (X * betas - y ) ...i.e X with x_i0 = 1
        m = number of samples
        '''
        # cost = (1/(2 * n_samples)) * (X * self.betas - y).transpose() * ( X * betas - y)
        n_samples, n_features = X.shape
        cost = 0
        for i in np.arange(n_samples):
            prediction = self.predict(X[i])
            # print 'Prediction=%f y=%f' % (prediction, y[i])
            error = prediction - y[i]
            squared_error = pow(error, 2)
            # print 'Error=%f' % squared_error
            cost += squared_error
        # cost = format((1/(2*n_samples)) * cost, '.10f')
        cost = cost/(2*n_samples)
        # print 'Cost={}'.format(cost)
        return cost

    def gradient_descent(self, X, y):
        ''' Batch gradient descent update '''
        n_samples, n_features = X.shape

        for j in np.arange(n_features):
            e = 0
            for i in np.arange(n_samples):
                e += (self.predict(X[i]) - y[i]) * X[i][j]
            try:
                self.beta[j] -= float((self.alpha * e) / n_samples)
            except Exception as e:
                self.beta[j] -= 0

        return self.beta


def plot_cost(costs, iterations, alphas):

    fig, axes = plt.subplots(2, 5)

    k = 0
    #print np.array([x]), np.array([costs[k]])

    for i in np.arange(axes.shape[0]):
        for j in np.arange(axes.shape[1]):
            x = np.arange(iterations[k])
            axes[i, j].plot(x, costs[k])
            axes[i, j].set_ylabel('Cost J')
            axes[i, j].set_xlabel('Number of iterations')
            axes[i, j].title.set_text('Alpha={}'.format(alphas[k]))

            k += 1

    plt.show()

def plot_decision_boundary(data, beta):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], zs=data[:, 2], zdir='z')
    ax.set_xlabel('Age(Years)')
    ax.set_ylabel('Weight(Kilograms)')
    ax.set_zlabel('Height(Meters)')

    lm = LinearRegression()
    lm.beta = beta

    predictions = []
    #X = np.zeros(data.)
    n_samples, n_features = data.shape
    for i in np.arange(n_samples):
        predictions.append(lm.predict(X[i]))

    predictions = np.array(predictions)

    #ax.plot(np.arange(5), np.arange(50), )
    plt.show()


def normalize(x, std, mean):
    ''' '''
    return (x - mean)/std


def scale_features(X):

    scale = np.vectorize(normalize)

    scaled_X = np.zeros((X.shape[0], 2))

    for i in range(0, 2):
        std = X[:, i].std()
        mean = X[:, i].mean()
        scaled_X[:, i] = scale(X[:, i], std, mean)

    return scaled_X

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python problem2.py <input_csv> <output_csv>'
        sys.exit()

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 1.2]
    max_iterations = 5

    betas = []
    costs = []
    iterations = []
    output_csv = open(output_csv, 'w')
    dataset = np.genfromtxt(input_csv, delimiter=',', dtype='float64')
    X = dataset[:, 0:2]
    X = scale_features(X)
    X = np.insert(X, 0, 1, axis=1)  # x0 = 1
    y = dataset[:, 2]

    for a in alphas:
        try:
            lm = LinearRegression()
            lm.alpha = a
            lm.max_iterations = max_iterations
            lm.fit(X, y)
            # print lm.beta_list[0]
            # print lm.cost_list[-6:-1]
            betas.append(lm.beta_list)
            costs.append(lm.cost_list)
            iterations.append(lm.iterations)
        except Exception as e:
            print e
            raise

    for i, a in enumerate(alphas):
        line = [a, iterations[i]]
        line.extend(betas[i][-1])
        #print costs[i][-1]
        line = map(lambda x: str(x), line)
        # print line
        line = ','.join(line)
        output_csv.write(line + '\n')
    output_csv.close()

    #plot_cost(costs, iterations, alphas)
    #plot_decision_boundary(np.insert(dataset, 0, 1, axis=1), betas[4])
