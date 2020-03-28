import random
import numpy as np
import string
import copy
import random
import math
import time
import pandas as pd
from datetime import datetime, timedelta


class DNA(object):
    genes = []
    fitness =  math.inf # Se selecciona inicialmente como mejor individuo el primero.
    def __init__(self, size):
        self.size = size
        self.genes = copy.deepcopy(self.__generateWeights(self.size)) #Llenamos con valores entre 0 y 1 que representan los procentajes a invertir para cada acción 
    
    def __generateWeights(self, size):
        weights = np.random.random(size)
        weights = weights / np.sum(weights)
        return weights
    
    def mutate(self,mutation_rate):
        
        for ix in range(self.size):
            if random.random() < mutation_rate:
                self.genes[ix] = random.uniform(0, 1)
        
        if sum(self.genes) != 1:
            self.genes = self.genes / np.sum(self.genes)

class Population(object):
    def __init__(self, inversion, assets, df, maxPop, mutation_rate):
        self.df = np.log(df/df.shift(1))
        self.size = len(assets)
        self.inversion = inversion
        self.assets = assets
        self.maxPop = maxPop
        self.historico_mejor_fitness = []
        self.historico_mejor_predictores = []
        self.diferencia_abs = []
        self.biggest = 0
        self.avg_fitness = 0
        self.mutation_rate = mutation_rate
        self.pop= []
    
    """
    Este método genera un nuevo individuo a partir de dos individuos
    parentales empleando el método de cruzamiento uniforme.
    """

    def cruce(self,parentA, parentB, metodo_cruce = "uniforme"):

        
        parental_1 = parentA
        parental_2 = parentB
        
        descendencia = copy.deepcopy(parentA)
        descendencia.genes = np.repeat(None, self.size)
        
        if metodo_cruce == "uniforme":
            # Se seleccionan aleatoriamente las posiciones que se heredan del
            # padre1 y del padre2.
            herencia_parent_1 = np.random.choice(
                                    a       = [True, False],
                                    size    = self.size,
                                    p       = [0.5, 0.5],
                                    replace = True
                                )
            herencia_parent_2 = np.logical_not(herencia_parent_1)

            # Se transfieren los valores al nuevo individuo.
            descendencia.genes[herencia_parent_1] \
                = parental_1.genes[herencia_parent_1]

            descendencia.genes[herencia_parent_2] \
                = parental_2.genes[herencia_parent_2]
            
        if metodo_cruce == "punto_simple":
            punto_cruce  = np.random.choice(a = np.arange(1, self.size - 1),size = 1)
            punto_cruce = punto_cruce[0]
            descendencia.genes = np.hstack(   
                                            (parental_1.genes[:punto_cruce],
                                            parental_2.genes[punto_cruce:])
                                        )
         # La suma de los pesos de los indicviduos debe sumar 1                                
        if sum(descendencia.genes) != 1:
            descendencia.genes = descendencia.genes / np.sum(descendencia.genes)
            
        return descendencia
        
    def generatePopulation(self):
        self.pop = [DNA(len(self.assets)) for i in range(self.maxPop)]
    
    def __sharpe(self, genes):
        # Expected return
        exp_ret = np.sum((self.df.mean()*genes)*252)
        # Expected volatility
        exp_vol = np.sqrt(np.dot(genes.T,np.dot(self.df.cov()*252, genes)))
        return exp_ret/exp_vol

    def __fitness(self, genes):
        return self.__sharpe(genes) * -1

    def calculateFitness(self):
        # Determine the fitness of an individual. Hight is better.
        
        self.biggest = 0
        self.second = 0
        self.avg_fitness = 0
        
        for ix in range(len(self.pop)):
            #Fitness score is the sum of the correct letters
            self.pop[ix].fitness = self.__fitness(self.pop[ix].genes) 
            
            self.avg_fitness += float(self.pop[ix].fitness) / len(self.assets)            
            #Save the 2 highest fitness for reproduction
            if self.pop[ix].fitness < self.pop[self.biggest].fitness:
                self.biggest = ix
            elif self.pop[ix].fitness < self.pop[self.second].fitness:
                self.second = ix

        #Calculate average fitness
        self.avg_fitness = (self.avg_fitness / len(self.pop)) * 100.0

    def nextGeneration(self):     
        #Obtenemos los genes de los mejores padres
        parentA =  self.pop[self.biggest]
        parentB =  self.pop[self.second]

        for ix in range(len(self.pop)):
            # Cruzar parentales para obtener la descendencia
            child = self.cruce(parentA, parentB,)
            # Mutar la descendencia
            child.mutate(self.mutation_rate)
            self.pop[ix] = child

    def optimization(self, mutation_rate = 0.1, n_generaciones = 10, parada_temprana = False, rondas_parada = 0, tolerancia_parada = 0.01,  metodo_cruce = "uniforme"):
        """
        Este método realiza el proceso de optimización de una población.
        Parámetros

        mutation_rate : `float`, optional
            probabilidad que tiene cada posición del individuo de mutar.
            (default 0.1)
            
        n_generaciones : `int`
            número de repeticiones para la validación. EL método empleado es 
            `ShuffleSplit` de scikit-learn. (default 10)

        parada_temprana : `bool`, optional
            si durante las últimas `rondas_parada` generaciones la diferencia
            absoluta entre mejores individuos no es superior al valor de 
            `tolerancia_parada`, se detiene el algoritmo y no se crean nuevas
            generaciones. (default ``False``)

        rondas_parada : `int`, optional
            número de generaciones consecutivas sin mejora mínima para que se
            active la parada temprana. (default ``None``)

        tolerancia_parada : `float` or `int`, optional
            valor mínimo que debe tener la diferencia de generacio
        
        metodo_cruce : {"uniforme", "punto_simple"}
            método de cruamiento empleado.
        """
        generation = 0
        start = time.time()
            


        # EVALUAR INDIVIDUOS DE LA POBLACIÓN
        # ------------------------------------------------------------------
        while generation in np.arange(n_generaciones):
            print("-------------")
            print("Generación: " + str(generation))
            print("-------------")
            self.calculateFitness() 
            
            # SE ALMACENA LA INFORMACIÓN DE LA GENERACIÓN EN LOS HISTÓRICOS
            # ------------------------------------------------------------------
            self.historico_mejor_fitness.append(copy.deepcopy(self.pop[self.biggest].fitness))
            self.historico_mejor_predictores.append(copy.deepcopy(self.pop[self.biggest].genes))
            
            
            
            # SE CALCULA LA DIFERENCIA ABSOLUTA RESPECTO A LA GENERACIÓN ANTERIOR
            # ------------------------------------------------------------------
            # La diferencia solo puede calcularse a partir de la segunda
            # generación.
            if generation == 0:
                self.diferencia_abs.append(None)
            else:
                diferencia = abs(self.historico_mejor_fitness[generation] \
                                - self.historico_mejor_fitness[generation-1])
                self.diferencia_abs.append(diferencia)
                
            # CRITERIO DE PARADA
            # ------------------------------------------------------------------
            # Si durante las últimas n generaciones, la diferencia absoluta entre
            # mejores individuos no es superior al valor de tolerancia_parada,
            # se detiene el algoritmo y no se crean nuevas generaciones.
            if parada_temprana and generation > rondas_parada:
                ultimos_n = np.array(self.diferencia_abs[-(rondas_parada): ])
                if all(ultimos_n < tolerancia_parada):
                    print("Algoritmo detenido en la generación " 
                        + str(generation) \
                        + " por falta cambio absoluto mínimo de " \
                        + str(tolerancia_parada) \
                        + " durante " \
                        + str(rondas_parada) \
                        + " generaciones consecutivas.")
                    break
            
            self.nextGeneration()
            #print("Generation:", generation)
            #print("Average fitness:", self.avg_fitness )
            generation += 1

        indice_valor_optimo  = np.argmax(np.array(self.historico_mejor_fitness))
        predictores_optimos = self.historico_mejor_predictores[indice_valor_optimo]
        valor_fitness_optimo= self.historico_mejor_fitness[indice_valor_optimo]

        resultados_df = pd.DataFrame(
            {
            "mejor_fitness"        : self.historico_mejor_fitness,
            "mejor_predictores"    : self.historico_mejor_predictores,
            "diferencia_abs"       : self.diferencia_abs
            }
        )

        end = time.time()     
        print("-------------------------------------------")
        print("Optimización finalizada " \
            + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("-------------------------------------------")
        print("Duración optimización: " + str(end - start))
        print("Número de generaciones: " + str(generation))
        print("Valor métrica óptimo: " + str(valor_fitness_optimo))
        print("Pesos óptimos: " + str(predictores_optimos))
        print("Cartera de acciones: " + str(self.assets))
        print("Rendimientos medio los valores: " + str(self.df.mean().values))
        print("")

        return resultados_df