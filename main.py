import Genetics
import Yahoo
import string
import numpy as np
import random
import numpy as np
import string
import copy
import random
import math
import time
import pandas as pd
import copy
from matplotlib import pyplot as plt
import time


def loadData(assets):
    df= pd.DataFrame()
    for a in assets:
        wft = Yahoo.Finance(a, days_back=360).get_quote()
        df_asset = wft["Adj Close"].to_frame(name=a)
        df = pd.concat([df_asset, df], axis=1, sort=False)
    df = df.dropna()
    return df

def main(assets = ["CLNX.MC","MTS.MC","FER.MC","SAN.MC"]):
    df = loadData(assets)
    maxPop = 500
    mutation_rate = 0.20
    n_generaciones = 100
    inversion = 10
    parada_temprana = True # si durante las últimas `rondas_parada` generaciones la diferencia absoluta entre mejores individuos no es superior al valor de  `tolerancia_parada`, se detiene el algoritmo
    rondas_parada = 10 #  número de generaciones consecutivas sin mejora mínima para que se active la parada temprana.
    tolerancia_parada = 0.01 #valor mínimo que debe tener la diferencia de generaciones consecutivas

    population = Genetics.Population(inversion, assets, df, maxPop, mutation_rate)
    
    population.generatePopulation()

    results = population.optimization(mutation_rate, n_generaciones, parada_temprana, rondas_parada, tolerancia_parada)

    population.printHistoric(n_generaciones)

if __name__ == '__main__':
    main()