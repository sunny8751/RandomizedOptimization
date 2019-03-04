import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def scoreWhiteWine(saveFig):
    # Score for wine dataset
    def lineValue():
        oas = ['GA']
        for name in oas:
            train = []
            test = []
            def openScores(arr, name, trainTest):
                with open("ABAGAIL/scores/{}_{}_scores.txt".format(name, trainTest), 'r') as f:
                    for line in f:
                        arr.extend([100*float(x) for x in line[:-1].split(",")])
            openScores(train, name, "train")
            openScores(test, name, "test")

            title = "{} Learning Curve".format(name)
            x = np.arange(len(train))

            plt.figure()
            plt.title(title)
            plt.plot(x, train, '-', label="Train score")
            plt.plot(x, test, '-', label="Test score")
            plt.legend()
            plt.xlabel("Number of Iterations")
            plt.ylabel('Accuracy Score (%)')

            if saveFig:
                plt.savefig("ABAGAIL/figures/white_wine/{}_lc.png".format(name))

            plt.show()


    def barTime():
        title = "White Wine Neural Network Optimization Times"
        x = ['RHC', 'SA', 'GA']

        times = []
        with open("ABAGAIL/scores/white_wine_times.txt", 'r') as f:
            for line in f:
                for alg in x:
                    if line.startswith(alg): times.append(float(line.split(": ")[1]))

        plt.figure()
        plt.title(title)
        plt.bar(x, np.log10(times))
        # plt.legend()
        plt.xlabel("Number of Colors")
        plt.ylabel('Logarithmic Time to Run (Log_10 (seconds))')

        if saveFig:
            plt.savefig("ABAGAIL/figures/white_wine/white_wine_times.png")

        plt.show()

    lineValue()
    barTime()

def travelingSalesman(saveFig):
    # Traveling Salesman
    dataVal = defaultdict(list)
    dataTime = defaultdict(list)
    with open("ABAGAIL/scores/traveling_salesman.txt", 'r') as f:
        i = 0
        for line in f:
            line = line.strip()
            if line == "": continue
            n, maxVal, time = line.split(",")
            n = int(n)
            time = float(time)/1000
            maxVal = float(maxVal)
            dataVal[n].append(maxVal)
            # dataTime[i % 4].append(time)
            dataTime[n].append(time)

            i += 1

    def lineValue():
        title = "Traveling Salesman"
        x = sorted(list(dataVal.keys()))
        rhc = [dataVal[a][0] for a in x]
        sa = [dataVal[a][1] for a in x]
        ga = [dataVal[a][2] for a in x]
        mimic = [dataVal[a][3] for a in x]

        plt.figure()
        plt.title(title)
        plt.plot(x, rhc, '-', label="RHC")
        plt.plot(x, sa, '-', label="SA")
        plt.plot(x, ga, '-', label="GA")
        plt.plot(x, mimic, '-', label="MIMIC")
        plt.legend()
        plt.xlabel("Number of Cities")
        plt.ylabel('1/Distance')

        if saveFig:
            plt.savefig("ABAGAIL/figures/traveling_salesman/traveling_salesman.png")

        # plt.show()

    def lineTime():
        title = "Traveling Salesman Times"

        plt.figure()
        plt.title(title)
        # x = ['RHC', 'SA', 'GA', 'MIMIC']
        # plt.bar(x, [sum(dataTime[a]) / len(dataTime[a]) for a in range(len(x))])
        # plt.legend()

        x = sorted(list(dataTime.keys()))
        rhc = [dataTime[a][0] for a in x]
        sa = [dataTime[a][1] for a in x]
        ga = [dataTime[a][2] for a in x]
        mimic = [dataTime[a][3] for a in x]
        print(mimic, np.log10(mimic))

        plt.plot(x, np.log10(rhc), '-', label="RHC")
        plt.plot(x, np.log10(sa), '-', label="SA")
        plt.plot(x, np.log10(ga), '-', label="GA")
        plt.plot(x, np.log10(mimic), '-', label="MIMIC")
        plt.legend()

        plt.xlabel("Number of Cities")
        plt.ylabel('Logarithmic Time to Run (Log_10 (seconds))')

        if saveFig:
            plt.savefig("ABAGAIL/figures/traveling_salesman/traveling_salesman_times.png")

        plt.show()

    lineValue()
    lineTime()

def maxKColoring(saveFig):
    # Max K Coloring
    dataSuccess = defaultdict(list)
    dataIter = defaultdict(list)
    dataTime = defaultdict(list)
    with open("ABAGAIL/scores/max_k_coloring.txt", 'r') as f:
        content = f.readlines()
        for i in range(len(content)/4):
            line1 = content[4*i].strip()
            line2 = content[4*i+1].strip()
            line3 = content[4*i+2].strip()

            alg, iter = line1.split(": ")
            iter = float(iter)
            success = line2.startswith("Found")
            time = float(line3.split(": ")[1])/1000
            # if not success: iter = 0
            dataSuccess[alg].append(success)
            dataIter[alg].append(iter)
            dataTime[alg].append(time)

    def barSuccess():
        title = "Max K Coloring"
        x = ['RHC', 'SA', 'GA', 'MIMIC']

        plt.figure()
        plt.title(title)
        plt.bar(x, [dataSuccess[a].count(True) for a in x])
        print([dataSuccess[a].count(True) for a in x])
        # plt.legend()
        plt.xlabel("Number of Colors")
        plt.ylabel('Percent of Successful Colorings (%)')

        if saveFig:
            plt.savefig("ABAGAIL/figures/max_k_coloring/max_k_coloring.png")

    def barTime():
        title = "Max K Coloring Times"
        x = ['RHC', 'SA', 'GA', 'MIMIC']

        plt.figure()
        plt.title(title)
        plt.bar(x, [sum(dataTime[a])/len(dataTime[a]) for a in x])
        # plt.legend()
        plt.xlabel("Optimization Algorithm")
        plt.ylabel('Time to Run (seconds)')

        if saveFig:
            plt.savefig("ABAGAIL/figures/max_k_coloring/max_k_coloring_times.png")

        plt.show()

    barSuccess()
    barTime()

def continuousPeaks(saveFig):
    # Continous Peaks
    dataValue = defaultdict(list)
    dataTime = defaultdict(list)
    with open("ABAGAIL/scores/continuous_peaks_T10.txt", 'r') as f:
        for line in f:
            if line.strip() == "": continue
            n, t, alg, maxVal, time = line.split(",")
            n = int(n)
            t = int(t)
            maxVal = float(maxVal)
            time = float(time)/1000
            dataValue[n].append(maxVal)
            dataTime[n].append(time)

    def lineValue():
        title = "Continuous Peaks (T=N/10)"
        x = sorted(list(dataValue.keys()))
        rhc = [dataValue[n][0] for n in x]
        sa = [dataValue[n][1] for n in x]
        ga = [dataValue[n][2] for n in x]
        mimic = [dataValue[n][3] for n in x]

        plt.figure()
        plt.title(title)
        plt.plot(x, rhc, '-', label="RHC")
        plt.plot(x, sa, '-', label="SA")
        plt.plot(x, ga, '-', label="GA")
        plt.plot(x, mimic, '-', label="MIMIC")
        plt.legend()
        plt.xlabel("N Value")
        plt.ylabel('Maximum Fitness Value')

        if saveFig:
            plt.savefig("ABAGAIL/figures/continuous_peaks/continuous_peaks_T10.png")

        plt.show()

    def lineTime():
        for n in dataValue.iterkeys():
            title = "Continuous Peaks Times (T=N/10)".format(n)
        x = sorted(list(dataTime.keys()))
        rhc = [dataTime[n][0] for n in x]
        sa = [dataTime[n][1] for n in x]
        ga = [dataTime[n][2] for n in x]
        mimic = [dataTime[n][3] for n in x]

        plt.figure()
        plt.title(title)
        plt.plot(x, np.log10(rhc), '-', label="RHC")
        plt.plot(x, np.log10(sa), '-', label="SA")
        plt.plot(x, np.log10(ga), '-', label="GA")
        plt.plot(x, np.log10(mimic), '-', label="MIMIC")
        plt.legend()
        plt.xlabel("N Value")
        plt.ylabel('Logarithmic Time to Run (Log_10 (seconds))')

        if saveFig:
            plt.savefig("ABAGAIL/figures/continuous_peaks/continuous_peaks_T10_times.png")

        plt.show()

    lineValue()
    lineTime()

# scoreWhiteWine(True)
# continuousPeaks(True)
# travelingSalesman(True)
# maxKColoring(True)
