import csv
import random
import math
import cmath
import operator

def calcAttr(dados = [], atributos = []):
    #cada coluna da matriz dados vai ser usada para fazer os cálculos de atributos
    result = float(0)
    c=0
    while c <6:
        val = []
        l=0
        while l<15:
            val.append(dados[l][c])
            l=l+1
        #primeiro atributo
        average(val, atributos)
        #segundo atributo e terceiro atributo
        derivative(val, atributos)
        #quarto atributo
        monotonicity(val, atributos)
        #transformada de fourier receberá 8 valores de amplitude para cada um dos 6 atributos dados por c
        fourier(val, atributos) #atributos tem 4*6+8*6=72 valores para cada instância
        c=c+1

def fourier(val = [], atributos=[]):
    v=[]
    N=15
    for i in range(1,9):
        v.append(i)
    k=0
    pi=round(cmath.pi,7)
    while k < 8:
        ak=[0,0,0,0,0,0,0,0]
        for n in range(15):
            ak[k]=ak[k]+(round(1.0/N,5))*(val[n])*cmath.exp(-v[k]*1j*(2*pi/N)*(n+1))
        ak[k]=ak[k]+(round(1.0/N,5))*(val[14])*cmath.exp(-v[7]*1j*(2*pi/N)*16)
        atributos.append(ak[k])
        k=k+1

def monotonicity(val = [], atributos = []):
    i=0
    mon=0
    while i<14:
        if val[i] < val[i+1]:
            mon = mon + 1
        i=i+1
    atributos.append(mon)
    
def derivative(val=[], atributos = []):
    der = val[0]-val[7]
    der = der/7
    deri = val[7] - val[14]
    deri = deri/7
    atributos.append(der)
    atributos.append(deri)

def average(val=[], atributos=[]):
    #média de : media tomada de 3 em 3, media de 5 em 5, media dos 15
    total=0.0
    i=0
    while i < 3:
        total=total+float(val[i])
        i=i+1
    avg = total/3
    total=0.0
    for i in range(3,6):
        total=total+float(val[i])
    avg = avg + (total/3)
    total=0.0
    for i in range(6,9):
        total=total+float(val[i])
    avg = avg + (total/3)
    total=0.0
    for i in range(9,12):
        total=total+float(val[i])
    avg = avg + (total/3)
    total=0.0
    for i in range(12,15):
        total=total+float(val[i])
    avg = avg + (total/3)

    total=0.0
    for i in range(5):
        total=total+float(val[i])
    avg = avg + (total/5)
    total=0.0
    for i in range(5,10):
        total=total+float(val[i])
    avg = avg + (total/5)
    total=0.0
    for i in range(10,15):
        total=total+float(val[i])
    avg = avg + (total/5)

    total=0.0
    for i in range(15):
        total=total+float(val[i])
    avg = avg + (total/15)
    avg = avg/9
    atributos.append(avg)
    
def loadDataset(filename = 'lp1.data', split = 0.67, trainingSet=[] , testSet=[], attr = 6):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile, delimiter="\t") 
        dataset = list(lines)
        #print (dataset, '\n\n')
        x = 0
        while x < (len(dataset)):
                dados =[]
                #primeira informação é a classe
                classe = str(dataset[x][0])
                #print ('Classe: ', classe)
                x = x+1
                #dados da classe
                d = 0
                y=1
                while d < 15:
                    linha = []
                    linha = dataset[x+d]
                    d= d+1
                    linha = linha[1:]
                    i=0
                    while i < len(linha):
                        linha[i] = int(linha[i])
                        i=i+1
                    
                    dados.append(linha)
                x = x+17
                #print ('Dados: \n', dados, '\n')
                #Da matriz dados faz os cálculos e retorna uma linha de instâncias
                atributos = []
                calcAttr(dados, atributos)
                atributos.append(classe)
                #print('Atributos: ', len(atributos), '\n')
                #print(atributos)
                #rand = random.random()
                #print('\n  random: ', rand)
                if random.random() < split:
                    trainingSet.append(atributos)
                else:
                    testSet.append(atributos)
                
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def euclideanDistance(instance1, instance2, length):
	distance = 0
	d=0.0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	d = math.sqrt(pow(distance.real,2)) + math.sqrt(pow(distance.imag,2)) #caso seja imaginario converte para real
	return math.sqrt(d)

def executaKNN(dataset='lp1.data'):
    # prepare data and use KNN to classify the instances
    trainingSet=[]
    testSet=[]
    split = 0.67
    attr = 6
    loadDataset(dataset, split, trainingSet, testSet, attr)
    print('\n\nUsing KNN in the ', dataset, ' database')
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    accuracy = round(accuracy,2)
    print('Accuracy: ' + repr(accuracy) + '%')

def main():
    executaKNN('lp1.data')
    executaKNN('lp2.data')
    executaKNN('lp3.data')
    executaKNN('lp4.data')
    executaKNN('lp5.data')

    
main()
