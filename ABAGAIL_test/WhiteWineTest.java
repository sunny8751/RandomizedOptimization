package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import util.linalg.DenseVector;
//import org.apache.commons;

import java.util.*;
import java.io.*;
import java.text.*;
import java.lang.reflect.Array;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */

public class WhiteWineTest {
    // private static int rowCount = 1599;

    private static Instance[] train_set = initializeInstances("./white-wine-train.csv", true);
    private static Instance[] test_set = initializeInstances("./white-wine-test.csv", true);
    private static Instance[] instances = concatenate(train_set, test_set);

    private static int inputLayer = 11, outputLayer = 10, trainingIterations = 1000;

    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    // private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];

    private static String[] oaNames = {"RHC", "SA", "GA"};
    // private static String[] oaNames = {"GA"};

    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) throws Exception {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    //new int[] {inputLayer, hiddenLayer, outputLayer}
                    // new int[] {inputLayer, hiddenLayer1, hiddenLayer2, outputLayer}
                    new int[] {inputLayer, (inputLayer + outputLayer) / 2, outputLayer}
                );
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        //Uncomment out this line for Randomized HIll Climbing
        // oa[0] = new RandomizedHillClimbing(nnop[0]);

        //Uncomment out this line for Simulating Annealing
//        oa[0] = new SimulatedAnnealing(1E11, .95, nnop[0]);

        //Uncomment out this line for Genetic Algorithms
         // oa[0] = new StandardGeneticAlgorithm(200, 100, 10, nnop[0]);

        //
        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            // train(oa[i], networks[i], oaNames[i]); //trainer.train();

            BufferedWriter trainBr = new BufferedWriter(new FileWriter("scores/" + oaNames[i] + "_train_scores.txt"));
            BufferedWriter testBr = new BufferedWriter(new FileWriter("scores/" + oaNames[i] + "_test_scores.txt"));

            for(int j = 0; j < trainingIterations; j++) {
                oa[i].train();

                Instance optimalInstance = oa[i].getOptimal();
                networks[i].setWeights(optimalInstance.getData());

                //Training set
                double trainScore = scoreTrain(i);

                //Testing set
                double testScore = scoreTest(i);

                trainBr.write(trainScore + ",");
                testBr.write(testScore + ",");
            }

            trainBr.close();
            testBr.close();

            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            // //Training set
            // scoreTrain(trainingTime, i);

            // //Testing set
            // scoreTest(trainingTime, i);
            System.out.println("Training time: " + df.format(trainingTime));
        }

        // System.out.println(results);i love erika sheng
    }

    private static double scoreTrain(int oaIndex) {
        double predicted, actual, start, end, testingTime, correct = 0, incorrect = 0;
        start = System.nanoTime();
        for(int j = 0; j < train_set.length; j++) {
            networks[oaIndex].setInputValues(instances[j].getData());
            networks[oaIndex].run();

            predicted = instances[j].getLabel().getData().argMax();
            actual = networks[oaIndex].getOutputValues().argMax();
            // System.out.println("predicted " + predicted + ", actual " + actual);

            //double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            if (predicted == actual) {
                correct++;
            } else {
                incorrect++;
            }

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        // results +=  "\nTraining Results for " + oaNames[oaIndex] + ": \nCorrectly classified " + correct + " instances." +
        //         "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
        //         + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
        //         + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

        // System.out.println("Train " + (correct / (correct + incorrect)));
        return correct / (correct + incorrect);
    }

    private static double scoreTest(int oaIndex) {
        double testingTime, start, end, predicted, actual, correct = 0, incorrect = 0;
        //BackPropagationNetwork cloned = new BackPropagationNetwork(networks[i]);
        start = System.nanoTime();
        for(int j = 0; j < test_set.length; j++) {
            networks[oaIndex].setInputValues(test_set[j].getData());
            networks[oaIndex].run();

            predicted = instances[j].getLabel().getData().argMax();
            actual = networks[oaIndex].getOutputValues().argMax();
            // System.out.println("predicted " + predicted + ", actual " + actual);

            //double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            if (predicted == actual) {
                correct++;
            } else {
                incorrect++;
            }

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);

        // results +=  "\nTest Results for " + oaNames[oaIndex] + ": \nCorrectly classified " + correct + " instances." +
        //         "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
        //         + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
        //         + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

        // System.out.println("Test " + (correct / (correct + incorrect)));
        return correct / (correct + incorrect);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(network.getOutputValues()));
                //error = 1;
                error += measure.value(output, example);
            }

            // System.out.println(df.format(error));
        }
    }

    public static int indexOfLargest(double array[])
    {
        int index = 0;
        double max_value = 0;

        for (int i = 0; i < array.length; i++) {
            if (array[i] > max_value) {
                index = i;
                max_value = array[i];
            }
        }

        return index;
    }

    private static Instance[] initializeInstances(String fileName, boolean header) {
        int numLines = 0;
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(fileName)));
            String line;
            while ((line = br.readLine()) != null) {
                numLines ++;
            }
            br.close();
        } catch(Exception e) {
            e.printStackTrace();
        }
        if (header) {numLines--;}

        double[][][] attributes = new double[numLines][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(fileName)));

            if (header) {
                br.readLine();
            }
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[11]; // 11 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 11; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
            br.close();
        } catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);

            // Read the digit 0-9 from the attribute array that was read from the csv
            int c = (int) attributes[i][1][0];

            int nClasses = 10;
            // Create a double array of length 10, all values are initialized to 0
            double[] classes = new double[nClasses];

            // Set the i'th index to 1.0
            classes[c] = 1.0;
            instances[i].setLabel(new Instance(classes));
        }

        return instances;
    }

   private static <T> T[] concatenate(T[] a, T[] b) {
        int aLen = a.length;
        int bLen = b.length;

        @SuppressWarnings("unchecked")
        T[] c = (T[]) Array.newInstance(a.getClass().getComponentType(), aLen + bLen);
        System.arraycopy(a, 0, c, 0, aLen);
        System.arraycopy(b, 0, c, aLen, bLen);

        return c;
    }
}
