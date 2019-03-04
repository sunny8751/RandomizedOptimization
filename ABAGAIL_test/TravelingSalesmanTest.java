package opt.test;

import java.util.Arrays;
import java.util.Random;
import java.io.*;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter("./scores/traveling_salesman.txt"));
        for (int h = 0; h < 10; h++) {
            N = 20 + h*10;
            Random random = new Random();
            // create the random points
            double[][] points = new double[N][2];
            for (int i = 0; i < points.length; i++) {
                points[i][0] = random.nextDouble();
                points[i][1] = random.nextDouble();
            }
            // for rhc, sa, and ga we use a permutation based encoding
            TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
            Distribution odd = new DiscretePermutationDistribution(N);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
            long starttime = System.currentTimeMillis();
            fit.train();
            writer.write(N + "," + ef.value(rhc.getOptimal()) + "," +  (System.currentTimeMillis() - starttime));
            writer.newLine();
            
            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            starttime = System.currentTimeMillis();
            fit.train();
            writer.write(N + "," + ef.value(sa.getOptimal()) + "," +  (System.currentTimeMillis() - starttime));
            writer.newLine();
            
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            starttime = System.currentTimeMillis();
            fit.train();
            writer.write(N + "," + ef.value(ga.getOptimal()) + "," +  (System.currentTimeMillis() - starttime));
            writer.newLine();
            
            // for mimic we use a sort encoding
            ef = new TravelingSalesmanSortEvaluationFunction(points);
            int[] ranges = new int[N];
            Arrays.fill(ranges, N);
            odd = new DiscreteUniformDistribution(ranges);
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            MIMIC mimic = new MIMIC(200, 100, pop);
            fit = new FixedIterationTrainer(mimic, 1000);
            starttime = System.currentTimeMillis();
            fit.train();
            writer.write(N + "," + ef.value(mimic.getOptimal()) + "," +  (System.currentTimeMillis() - starttime));
            writer.newLine();

        }

        writer.close();
    }

}