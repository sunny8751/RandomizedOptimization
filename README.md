# Analysis of Randomized Optimization Algorithms

Forked from [here](https://github.com/nirave/random-optimization-algorithms).

## Setup instructions

1. Download ABAGAIL from [here](https://github.com/pushkar/ABAGAIL) into this github root folder.

2. Copy files in "ABAGAIL_root" folder to the "ABAGAIL/" root folder you just downloaded.

3. Copy files in "ABAGAIL_test" folder to "ABAGAIL/src/opt/test/" folder.

4. Rebuild ABAGAIL project using ANT


## Run instructions
Note: After changing any code, rerun ANT to rebuild project before running.

1. To run the three algorithms (RHC, SA, GA) on the white wine dataset, run the command: "java -cp ABAGAIL.jar opt.test.WhiteWineTest"

You can also edit the java file to run individual algorithms on the wine dataset instead of all three at once. Remember to rebuild using ANT.

2. To run all four algorithms (including MIMIC) on the Traveling Salesman problem, run the command: "java -cp ABAGAIL.jar opt.test.TravelingSalesmanTest".

3. To run all four algorithms (including MIMIC) on the Continuous Peaks problem, run the command: "java -cp ABAGAIL.jar opt.test.ContinuousPeaksTest".

4. To run all four algorithms (including MIMIC) on the Max K Coloring problem, run the command: "java -cp ABAGAIL.jar opt.test.MaxKColoringTest".

5. To create the figures and graphs for the data, see plot.py. Uncomment the corresponding functions and run. The outputted figures will be saved in the "ABAGAIL/figures/.../" directory.
