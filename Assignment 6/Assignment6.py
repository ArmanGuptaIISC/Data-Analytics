import os 
import pandas as pd
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import svds
import numpy as np
import time
from tqdm import tqdm
from sys import getsizeof
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Preprocessing of the data 
## 1. Creating unique ids for each user staring from zero
## 2. Creating unique item id for each movie staring from zero

path = './data/ratings.csv'
data = pd.read_csv(path)
data['userId'] -= min(data['userId'])
rating, users, movieId = data['rating'].values , data['userId'].values , data['movieId'].values
unique_users , unique_movieId = np.unique(users) , np.unique(movieId)
def map_movie(mapper , m):
    return mapper[m]
movie_mapper = dict(zip(unique_movieId , np.arange(len(unique_movieId))))
data['movie_map'] = [map_movie(movie_mapper , m)  for m in movieId] 
movies = data['movie_map'].values

assert set(np.arange(max(movies)+1)).difference(np.unique(movies)) == set()
assert set(np.arange(max(users)+1)).difference(np.unique(users)) == set()

## creating sparse matrix to reduce the storage cost
um_rating_sparse = csc_matrix((rating, (users,movies)))



#######################################  SVD ###########################################
u2, s2, vT2 = svds(um_rating_sparse, k=min(max(users),max(movies)))
Total_Variance = np.sum(s2 ** 2) 
print('Total Variance Of the Data : ', Total_Variance)

## Want to preserve 0.9 * Total_Variance 
target_var = 0.9 * Total_Variance
cur_var = 0
k = len(s2) 
while cur_var < target_var:
    cur_var += s2[k-1] ** 2
    k -= 1
print(f'Reached Min Target_Var of {target_var} achieved using k = {len(s2) - k} largest singular values')

## Calculating SVD for each number of latent factors starting from 10 till min(max(users),max(movies)))
## This code is required for plotting the Error, Time , Space against number of latent factors in case of SVD 
svd_error , svd_time , svd_space , svd_var = [] ,[], [] ,[]
for i in tqdm(range(10,min(max(users),max(movies)))):
    begin = time.time()
    u,s,vT = svds(um_rating_sparse, k= i)
    end = time.time()
    A = np.dot(np.dot(u , np.diag(s)) , vT)
    fobeinus = np.linalg.norm(um_rating_sparse - A , ord = 'fro') 
    svd_error.append(fobeinus)
    svd_time.append(end - begin)
    svd_space.append(getsizeof(u) + getsizeof(s) + getsizeof(vT))
    svd_var.append(np.sum(s ** 2))


#######################################  CUR Decomposition ####################################
class CUR_Decomposition:
    def __init__(self, A):
        ''' 
            A - np.matrix()
        '''
        self.A = A
        self.rows , self.cols = A.shape

        ## Calculating Probability of picking columns and rows based on their energy
        self.A_A = np.multiply(A,A) # Hadamard product
        self.energy = np.sum(self.A_A)
        self.row_prob = np.array(np.sum(self.A_A , axis = 1) / self.energy).squeeze()
        self.col_prob = np.array(np.sum(self.A_A , axis= 0) / self.energy).squeeze()
        assert len(self.col_prob)  == A.shape[1]
        assert len(self.row_prob) == A.shape[0]
        
    def fit(self, k):
        C = np.matrix(np.zeros((self.rows,k)))
        R = np.matrix(np.zeros((k,self.cols)))

        ## Selecting K columns 
        count = 0
        a = np.arange(self.cols)
        visited_cols = []
        freq_visited_cols = []
        while count < k:
            col = np.random.choice(a  , p = self.col_prob)
            if col in visited_cols:
                index = visited_cols.index(col)
                freq_visited_cols[index] +=1 
            else:
                C[:,count] = self.A[:,col]  / np.sqrt(self.col_prob[col] * k)
                visited_cols.append(col)
                freq_visited_cols.append(1)
                count+=1
        C = np.multiply(C , np.sqrt(freq_visited_cols))

        ## Selecting K rows
        count = 0
        a = np.arange(self.rows)
        visited_rows = []
        freq_visited_rows = []
        while count < k:
            row = np.random.choice(a  , p = self.row_prob)
            if row in visited_rows:
                index = visited_rows.index(row)
                freq_visited_rows[index] +=1 
            else:
                R[count, : ] = self.A[row ,: ]  / np.sqrt(self.row_prob[row] * k)
                visited_rows.append(row)
                freq_visited_rows.append(1)
                count+=1
        R = np.multiply(R.T , np.sqrt(freq_visited_rows)).T


        ## Creating W from the intersections of selected rows and columns
        entry , rows , cols = [] ,[] ,[]
        for i,row  in enumerate(visited_rows):
            for j,col in enumerate(visited_cols):
                entry.append(self.A[row,col])
                rows.append(i)
                cols.append(j)

        W = csc_matrix((entry,(rows,cols)))
        X , sig , YT = svds(W)

        ## Finding U
        U = np.dot(np.dot(YT.T , np.diag(1/sig)) , X.T)
        return C , U , R

curs = CUR_Decomposition(A = um_rating_sparse.todense())

## Calculating CUR Decomposition for each number of latent factors starting from 10 till min(max(users),max(movies)))
## This code is required for plotting the Error, Time , Space against number of latent factors in case of CUR Decomposition 
cur_error , cur_time , cur_space = [] ,[], []
for i in tqdm(range(10,min(max(users),max(movies)))):
    begin = time.time()
    C, U, R = curs.fit(k = i)
    end = time.time()
    A = np.dot(np.dot(C , U) , R)
    fobeinus = np.linalg.norm(um_rating_sparse.todense() - A , ord = 'fro') 
    cur_error.append(fobeinus)
    cur_time.append(end - begin)
    cur_space.append(getsizeof(C) + getsizeof(U) + getsizeof(R))

## Creating Plots 
plt.figure(1)
plt.plot(np.arange(20 , 609) , svd_error[10:] , label = 'SVD ERROR')
plt.plot(np.arange(20 , 609) ,cur_error[10:] , label = 'CUR ERROR')
plt.legend()
plt.xlabel('Number of Latent Factors')
plt.ylabel('Error')
plt.title('Error VS Number of Latent Factors')
plt.savefig('./Plots/error_vs_lf.png')

plt.figure(2)
plt.plot(np.arange(20 , 609) ,svd_time[10:] , label = 'SVD')
plt.plot(np.arange(20 , 609) ,cur_time[10:] , label = 'CUR')
plt.legend()
plt.xlabel('Latent Factors')
plt.ylabel('Time (seconds) ')
plt.title('Time VS Number of Latent Factors')
plt.savefig('./Plots/time_vs_lf.png')

plt.figure(3)
plt.plot(np.arange(20 , 609) , svd_space[10:] , label = 'SVD')
plt.plot(np.arange(20 , 609) , cur_space[10:] , label = 'CUR')
plt.legend()
plt.title('Storage Required vs Number of Latent Factors')
plt.xlabel('Number of Latent Factors')
plt.ylabel('Space (Bytes)')
plt.savefig('./space_vs_lf.png')

plt.figure(4)
delta_error = np.array(cur_error) - np.array(svd_error)
plt.plot(np.arange(20 , 609) ,delta_error[10:] , label = 'CUR(error) - SVD(error)')
plt.legend()
plt.xlabel('Latent Factors')
plt.ylabel('Time (seconds) ')
plt.title('Additional Error Introduced by CUR Depcomposition as opposed to SVD')
plt.savefig('./Plots/addition_error_vs_lf.png')


######################################### PQ decomposition ###################################

train, test = train_test_split(data , test_size= 0.2 , random_state= 0)
train = train[['userId' , 'movie_map' ,'rating']]
test = test[['userId' , 'movie_map' ,'rating']]

class PQDecomposition:
    def __init__(self, lr, reg , k, epochs ):
        self.lr = lr 
        self.lam = reg   
        self.epochs = epochs
        self.k = k

    def fit(self, train):
        n_users = train['userId'].max()+1
        n_items = train['movie_map'].max()+1
        P = np.random.normal(0,.1,(n_users,self.k))
        Q = np.random.normal(0,.1,(n_items,self.k))
        
        for epoch in tqdm(range(self.epochs)):
            loss = 0
            for u,i,r in train.to_numpy():
                u,i = int(u) , int(i)
                dloss =  ( r - np.dot(P[u],Q[i]))
                deltaP = (-2* dloss * Q[i,:]  + 2 * self.lam * P[u,:])
                deltaQ = (-2* dloss * P[u,:] + 2 * self.lam *  Q[i,:])
                P[u,:] = P[u, :]- self.lr * deltaP
                Q[i,:] = Q[i ,:]- self.lr * deltaQ
                loss += dloss**2
            
            # if epoch % 10 == 0:
            print(f'Epochs {epoch} || loss {loss}')

        print('final Train loss : ' , loss)       
        self.P = P
        self.Q = Q
        self.n_users , self.n_items = n_users , n_items
        self.mean_rating = train['rating'].mean()
    
    def predict(self,u,i):
        u,i = int(u) , int(i)
        if u >= self.n_users or i >= self.n_items:
            return self.mean_rating
        return np.dot(self.P[u,:],self.Q[i,:])

print('Training PQ Decomposition Model')
model = PQDecomposition(lr = 1e-2, epochs=10, k = 200 , reg=1e-1)
model.fit(train)

pred_r = []
loss = 0
for u,i,r_ui in test.to_numpy():
    u , i = int(u) , int(i)
    p = model.predict(u,i)
    pred_r.append(p)
    loss += (p - r_ui) ** 2

print('TESTING PQ Decomposition Model')
print('Square Loss : ',loss )
print('RMSE : ',mean_squared_error(test['rating'].to_numpy() , pred_r))



##################################### Neural Collaborative Filtering Approach  #########################

n_users , n_items = data['userId'].nunique() , data['movie_map'].nunique()
U = pd.get_dummies(data['userId'] , columns= ['userId']).to_numpy()
I = pd.get_dummies(data['movie_map'] , columns= ['movie_map']).to_numpy()
ratings = data['rating'].to_numpy()
train_U , test_U , train_I , test_I , train_rating , test_rating = train_test_split(U,I,ratings, test_size = 0.2 ,random_state= 0)

## Implemented NCF from scratch; Reference - https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
class Embedding:
    def __init__(self , input_size , output_size):
        super().__init__()
        self.weights = np.random.normal(size=(input_size, output_size))
    
    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input , self.weights)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        return input_error

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.normal(size = (input_size, output_size) )
        self.bias = np.random.normal(size = (1, output_size) )

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        return input_error

class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
        
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def sigmoid_prime(x):
    sig = sigmoid(x)
    return  sig * ( 1 - sig )

def se(y_true, y_pred):
    ## Square Error 
    return np.sum(np.power(y_true-y_pred, 2))

def se_prime(y_true, y_pred):
    return 2*(y_pred-y_true)


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    

    def fit(self, U, I, y_train, epochs, learning_rate):
        samples = len(U)
        
        # training loop
        for i in range(epochs):
            err = 0
            # Batching with batch size 32
            for j in tqdm(range(0,samples,32)):
                # forward propagation
                user = U[j: j + 32]
                item = I[j: j + 32]
                output1 = self.layers[0].forward_propagation(user)
                output2 = self.layers[1].forward_propagation(item)
                output = output1 + output2
                for layer in self.layers[2:]:
                    output = layer.forward_propagation(output)
                
                # compute loss 
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in self.layers[::-1][:-2]:
                    error = layer.backward_propagation(error, learning_rate)
                
                self.layers[1].backward_propagation(error , learning_rate)
                self.layers[0].backward_propagation(error, learning_rate)
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

            
    # predict output for given input
    def predict(self, users, items):
        samples = len(users)
        result = []

        # run network over all samples
        for i in range(samples):
            output1 = self.layers[0].forward_propagation(users[i])
            output2 = self.layers[1].forward_propagation(items[i])
            output = output1 + output2
            for layer in self.layers[2:]:
                output = layer.forward_propagation(output)
            result.append(output)

        return np.array(result).squeeze()


embedding_dim = 200
net  = Network()
net.add(Embedding(n_users , embedding_dim))
net.add(Embedding(n_items , embedding_dim))
net.add(LinearLayer(embedding_dim , 50))
net.add(ActivationLayer(tanh,tanh_prime))
net.add(LinearLayer(50,25))
net.add(ActivationLayer(tanh,tanh_prime))
net.add(LinearLayer(25,1))
net.use(se, se_prime)

print('\n\nTraining NCF Model')
net.fit(train_U, train_I , train_rating, epochs = 10 ,learning_rate= 0.0001)

print('Testing NCF Model')

pred_r = net.predict(test_U , test_I)
loss = se(np.array(pred_r).squeeze() ,test_rating)
print('Square Loss : ', loss )
print('RMSE : ',mean_squared_error(test_rating, np.array(pred_r).squeeze()))
