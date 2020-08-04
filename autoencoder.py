import math
import numpy as np
import matplotlib.pyplot as plt

T = 4.0
nb_simu = 4000
nb_time = 400
delta = 20
nb_t_support = 20
nu = .35
Y0 = .6
dt = T / float(nb_time)
dt_sqrt = math.sqrt(dt)
dW_t = np.random.normal(size=(nb_simu,nb_time)) * dt_sqrt
beta = 0.8

Y = np.zeros((nb_simu, nb_time))
t = np.arange(0, T, dt)

for i in range(0, nb_simu):
    Y[i][0] = Y0
  
for i in range(0, nb_simu):
    for j in range(1, nb_time):
        Y[i][j] = Y[i][j-1] - beta * Y[i][j-1] * dt + nu * dW_t[i, j]

'''        
for i in range(0, 20):
    plt.plot(Y[i], color='red')
    
plt.show()
'''

mean_emp = np.zeros((nb_time))
var_emp = np.zeros((nb_time))

var_theo = nu * nu / (2.0 * beta) * (1.0 - np.exp(- 2.0 * beta * t))

for i in range(0, nb_time):
    mean_emp[i] = np.mean(Y[:,i])
    var_emp[i] = np.var(Y[:,i])

''' 
plt.plot(mean_emp, color='red', label='mean emp')
plt.legend()
plt.ylim(-0.05, 0.05)
plt.show()

plt.plot(var_emp, color='blue', label='var emp')
plt.plot(var_theo, color='green', label='var theo')
plt.legend()
plt.ylim(0, 1.1 * max(var_theo))
plt.show()
'''

V = np.zeros((nb_simu,nb_t_support-1))
for i_t in range(1, nb_t_support):
    V[:,i_t-1] = Y[:,i_t*delta]

cov = np.cov(V.T)

print(np.linalg.det(cov), np.linalg.cond(cov))

from numpy import linalg

eigen_values, eigen_vectors = linalg.eig(cov)

eigen = [(eigen_value, eigen_vectors[:,i]) for i, eigen_value in enumerate(eigen_values)]
eigen.sort(key=lambda pair: pair[0], reverse=True)

eigen_values, eigen_vectors = np.array([_eigen[0] for _eigen in eigen]), np.array([_eigen[1] for _eigen in eigen])

plt.plot(eigen_values, color='red')
plt.show()

#for _eigen_vector in eigen_vectors :
#    print(linalg.norm(_eigen_vector))


#test
'''
for i, _v in enumerate(eigen_vectors):
    _lambda = eigen_values[i]
    test = cov.dot(_v) - _lambda * _v
    _lambda_test = _v.T.dot(cov.dot(_v))
    print(_lambda_test, _lambda)
'''

A = eigen_vectors.T

#determine new coordinate representation

#v_simu = x_simu,i * e_i
#v_simu = x_simu,i * A * A_inv * e_i
#A_inv * e_i -> coodinates of new i-th basis vector expressed with resepct to standard basis e_1, ..., e_n
#x_tilde_simu, i = x_simu,i * A -> i-th coordinate of v_simu with respect to new basis given by A_inv * e_i

V_tilde = np.zeros((nb_simu,nb_t_support-1))

#eige_vectors is orthogonal, so A_T = A_inv

for i, _V in enumerate(V):
    V_tilde[i] = A.T.dot(_V)

'''
#test
for i, _Y in enumerate(Y):
    print(Y_tilde[i].dot(np.linalg.inv(eigen_vectors)) - Y[i])


#compare variances
for i in range(0, nb_t_support-1-1):
    for j in range(0, nb_t_support-1-1):
        if i == j:
            _var = np.var(V_tilde[:,i])
        elif i < j:
            _cov = np.cov(V_tilde[:,i], V_tilde[:,j])[0][1]
            assert abs(_cov) < 1e-10, 'deviation too big'
        else:
            continue
'''

plt.plot(A.T[0], color='red', label='1st')
plt.plot(A.T[1], color='green', label='2nd') 
plt.plot(A.T[2], color='blue', label='3rd') 
plt.legend()
plt.show()






relevant_motions = np.array([g])

class Proj:
    def __init__(self, subspace):
        self._subspace = subspace
        self._proj = subspace.dot(subspace.T)

    def __call__(self, v):
        return self._proj.dot(v)

proj = Proj(eigen_vectors)

print(proj)
