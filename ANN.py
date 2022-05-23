import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
import random
import pandas as pd


def sigmoid(x, y, b, val):    #Função de ativação dos neurônios que nao sejam da camada final

    z = np.array(np.dot(y, x),dtype=np.float32)
    
    if val:
        z = z.transpose() + b
    else:
        z += b

    a = (1/(1+np.exp(-z)))
            
    #a = np.array(a)

    return a, z

def softmax(x, y, b, val):    #Função de ativação para a camada final

    z = np.dot(y, x)
    
    if val:
        output = []
        z = z.transpose() + b
        for exp in np.exp(z):
            output.append(exp / np.sum(exp))
            len(exp)
        np.array(output)
    else:
        z += b
        exp = np.exp(z)
        output = exp / np.sum(exp)

    return output, z

def tanh(x, y, b):    # Outra função de ativação, que não foi usada até o momento
    z = np.dot(y, x) + b
    a = []
    for h in z:
        if h >= 700:
            a.append(1)

        elif h <= -700:
            a.append(0)

        else:
            a.append(np.tanh(h))
            
    a = np.array(a)

    return a, z

def derivate_sigmoid(X):  #Derivada da função de ativação sigmoid

    exp = np.exp(-X)
    
    der = (1/(1 + exp) * (1 - 1/(1 + exp)))

                       
    return der


def derivate_tanh(X):  #Derivada da função de ativação tangente
    der = []
    for x in X:
        
        if x < 700 and x > -700:   # Condição usada para não dar erro, mas espera-se que os valores nunca estejam fora desse intervalo, já que o neurônio ficaria saturado
            der.append(1-np.tanh(x)**2)
                       
        else:
            der.append(0)
                       
    return np.array(der)




class Network():
    def __init__(self, architecture):   # Aqui, serão iniciados aleatóriamente os parâmetros da rede, conforme estrutura informada em "architeture"
        self.architecture = architecture
        self.w = []
        self.b = []
        for i in np.arange(1,len(architecture)):
            self.w.append([])
            self.b.append([])
            for j in range(architecture[i]):
                self.b[i-1].append(np.random.normal(0,1))
                self.w[i-1].append([])
                for k in range(architecture[i-1]):
                    self.w[i-1][j].append(np.random.normal(0,1/np.sqrt(architecture[i-1])))  # Este desvio padrão é usado para previnir que os neurônios estejam saturados no começo
            self.w[i-1] = np.array(self.w[i-1])
            self.b[i-1] = np.array(self.b[i-1])
            
    def cost_function(self, x, y):    # Função de custo usada para trabalhar com a função de ativação softmax
        
        n = len(y)
        a, trash = self.evaluate_2(np.array(x))
        a = a[-1]
        
        #(np.array(y) - a)**2
        C = -np.sum(np.array(y) * np.log(np.array(a))) / (n)

        return C

    def quadratic_cost_function(self, x, y):    # função de custo quadratica -- uma opção para avaliar a rede.
        
        n = len(y)
        a, trash = self.evaluate_2(np.array(x))
        a = a[-1]
        
        #(np.array(y) - a)**2
        C = np.sum((np.array(y) - np.array(a))**2) / (2*n)

        return C

    def evaluate(self, y):      # Avalia a rede neural -- Recebe valores de entrada "y" e devolve a resposta da rede
        a = [y]                 # Aqui é avaliado a resposta da rede para um único dado de treinamento
        z = []
        for i in range(len(self.w)):
            
            if i == len(self.w) - 1:
                vec_aux = softmax(a[i], self.w[i], self.b[i], False)
            else:
                vec_aux = sigmoid(a[i], self.w[i], self.b[i], False)
                
            z.append(vec_aux[1])                  
            a.append(vec_aux[0])

            
        return a, z                # A saída mais importante é o "a", mas o "z" foi acrescentado aqui em uma tentativa de otimização para o resto do código
    
    
    def evaluate_2(self, y):      # Avalia a rede neural -- Recebe valores de entrada "y" e devolve a resposta da rede
        a = [np.array(y)]         # Aqui, o código inclui todos os dados de treinamento juntos, para aumentar a eficiência computacional
        z = []
        for i in range(len(self.w)):
            
            if i == len(self.w) - 1:
                vec_aux = softmax(a[i].transpose(), self.w[i], self.b[i], True)
            else:
                vec_aux = sigmoid(a[i].transpose(), self.w[i], self.b[i], True)
                
            z.append(vec_aux[1])                  
            a.append(vec_aux[0])

            
        return a, z               


    def update(self, y, x, learning_rate, nu):           # Função usada para atualizar os pesos, W's, e bias, b's (Gradient Descendent Method)
        
        eta = -learning_rate  # Velocidade de aprendizado
        
        #------------------------------------------------------------------------------------------------
        
        n = len(y)


        dw = []
        db = []
        
        for i in range(len(self.w)):
            dw.append(0)
            db.append(0)

        a, z = self.evaluate_2(np.array(y))  # Avalia os dados de entrada na rede e devolve os dados de saída, "a", e o "z", que é usado no resto do algoritmo   
                
        for l in range(n):           
            
            for i in range(len(self.w)):   # Rodando em todas as camadas da rede
                
                i = len(self.w) - i - 1    # O algorítmo do 'Back Propagation' começa no fim e termina no começo
                
                da_aux = 0


                d_sigma = derivate_sigmoid(z[i][l]) 

  
                '''
                ---------------------------------------------------------------------------------------------------------------------
                Aqui é calculado a derivada da função de custo em relação aos a's - matriz representando a saída de cada neurônio da rede
                
                '''

                if i == len(self.w) - 1 :
                    
                    da_aux = np.dot((a[i + 1][l] - x[l]), self.w[i])
                        
                else:
                    
                    da_aux = np.dot(da * d_sigma, self.w[i])
                    
                '''
                ---------------------------------------------------------------------------------------------------------------------
                Aqui é calculado a derivada da função de custo em relação aos w's - pesos dos neurônios
                
                '''
                        

                if i == len(self.w) - 1 :
                    
                    dw_aux = np.outer((a[i + 1][l] - x[l]), a[i][l]) + nu * self.w[i]

                else:
                    
                    dw_aux = np.outer(da * d_sigma, a[i][l]) + nu * self.w[i] 
                    
                '''
                ---------------------------------------------------------------------------------------------------------------------
                Aqui é calculado a derivada da função de custo em relação aos b's
                
                '''

                if i == len(self.w) - 1:              

                    db_aux = (a[i + 1][l] - x[l])
                    
                else:      
                    db_aux = da * d_sigma
                    
                # O resultado é, então, acresentado em dw e db, representando o passo a ser dado naquele instante

                dw[i] += eta * np.array(dw_aux)
                db[i] += eta * np.array(db_aux)
                da = da_aux
  
        #------------------------------------------------------------------------------------------------
            
        for m in range(len(self.w)):   # Os w's e b's são atualizados
            
            self.w[m] = self.w[m] + dw[m] / n
            self.b[m] = self.b[m] + db[m] / n

                    
        return None
    
    def run_epoch(self, learning_rate, nu, times_interaction, Training_input, training_output, waiting_time):    # Aqui é definido uma interação/epoch do algoritmo de otimização
        eta = learning_rate                     # Taxa de aprendizagem (Learning rate)
                                                # "nu" corresponde ao parâmetro de regularização
        minibatch = 200            # length of the minibatch. If 0, all the integer batch is going to be used

        if minibatch != 0:    # Define se é desejado usar o stochastic gradient descendent
            dataset = {}
            dataset['input'] = Training_input
            dataset['output'] = training_output
            dataset = pd.DataFrame(dataset)
            index = np.arange(0,len(Training_input)).tolist()


        start = time()
        for i in range(times_interaction):

            if minibatch != 0:  #Random choice do minibatch

                chose_index = random.sample(index, minibatch)
                Training_input_aux = dataset.loc[chose_index, 'input'].values.tolist()
                training_output_aux = dataset.loc[chose_index, 'output'].values.tolist()
                self.update(Training_input_aux, training_output_aux, eta, nu)

            else:    

                self.update(Training_input, training_output, eta, nu)    #Atualizamos os pesos e bias

            if i == 0 and waiting_time:           # Informa o tempo que levará para acabar
                stop = time()
                print(str(-start * times_interaction/3600 + stop * times_interaction/3600) + 'h para finalizar')

        return None
    
    def run(self, Training_input, training_output, learning_rate, epochs, regularization_parameter, tolerance):  # Esqueleto do algoritmo -- o algoritmo calcula o número de epochs desejadas, mas inclui um critério de parada, casos as novas interações se tornem irrelevantes

        c_1 = self.cost_function(Training_input, training_output)
        t = 0
        start = time()
        delta = c_1
        while delta >= tolerance and t <= epochs:    # Há uma tolerância mínima de diminuição da função de custo para considerar que ainda é útil continuar o algoritmo
            self.run_epoch(learning_rate, regularization_parameter, 20, Training_input, training_output, False)
            c_2 = self.cost_function(Training_input, training_output)
            delta = c_1 - c_2
            c_1 = c_2
            if t == 0:
                stop = time()
                print(str((stop - start) * (epochs/ 20) / 3600) + 'h para finalizar')
            t += 20

        if t < epochs:
            print('The NN converged with ' + str(t) + ' epochs')

        else:
            print('The NN did not converge') 
        return None
        
def test(net, validation_input, validation_output):  #Testa a acurácia da rede
    cont = 0
    result_output, trash = net.evaluate_2(np.array(validation_input))
    for i in range(len(validation_input)):
        
        test = interpretation(result_output[-1][i], validation_output[i])

        if test:
            cont += 1
    return cont/len(validation_output)*100


def saving(net):     #Salva os parâmetros da rede em um arquivo de texto
    for i in range(len(net.w)):
        file = open(str(i) + '.txt', 'w')
        for j in range(len(net.w[i])):
            for k in range(len(net.w[i][j])):
                file.write(str(net.w[i][j][k]) + ';')
            file.write('\n')




        file.close()
    file = open('b' + '.txt', 'w')
    for i in range(len(net.b)):
        for j in range(len(net.b[i])):
            file.write(str(net.b[i][j]) + ';')
        file.write('\n')
    file.close()
    
    return None

def recovering(net):    # Recupera os parâmetros da rede de um arquivo de texto
    for i in range(len(net.w)):
        file = open(str(i) + '.txt', 'r')
        line = file.readline()
        j = 0
        while line != '':
            aux = ''
            k = 0
            for c in line:
                if c != ';' and c != '\n':
                    aux += c
                elif c != '\n':
                    net.w[i][j][k] = float(aux)
                    aux = ''
                    k += 1
            j += 1
            line = file.readline()


    file = open('b' + '.txt', 'r')
    line = file.readline()
    i = 0
    while line != '':
        aux = ''
        j = 0
        for c in line:
            if c != ';' and c != '\n':
                aux += c
            elif c != '\n':
                net.b[i][j] = float(aux)
                aux = ''
                j += 1
        i += 1
        line = file.readline()
    return None


def interpretation(y, x):   # Compara a saída da rede com o desejado
    t = 0
    length = abs(y[0] - 1)
    for b in range(len(y)):
        if abs(y[b] - 1) < length:
            t = b
            length = abs(y[b] - 1)
            
    k = np.zeros(10)
    k[t] = 1
    aux = 1
    for i in range(len(x)):
        if x[i] != k[i]:
            aux = 0
        
    return aux

