import java.util.Arrays;

import javax.sound.sampled.SourceDataLine;

public class Perceptron {
    private static double error = 0;
    private static double m = 0.3; // Learning Constant
    private static double d = 0; // deltaWeight
    private static double sumError = 1; // sum_error
    private static double someValue = 0.05; // some_value
    private static double previousValue = 2; // some_value
    private static double[][] trainDataAND = { { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };
    private static int[] targetOutputsAND = { 0, 0, 0, 1 };
    private static double[][] trainDataOR = { { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };
    private static int[] targetOutputsOR = { 0, 1, 1, 1 };
    private static double[][] trainDataLinear = { { 0, 0, 1 }, { 1, 1, 1 }, { 2, 2, 1 }, { 3, 3, 1 }, { 0, 1.1, 1 },
            { 1, 2, 1 }, { 2, 3, 1 }, { 3, 4, 1 }, { 0, -1, 1 }, { 1, 0, 1 }, { 2, 1, 1 }, { 3, 2, 1 },
            { 0, -2, 1 }, { 1, -1, 1 }, { 2, 0, 1 }, { 3, 1, 1 } , { 3, 122, 1 } , { 3, 6.5, 1 } , { -12, -46.5, 1 },{-50,-101,1},{100,202,1}};
    private static int[] targetOutputsLinear = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1,0,0,0,1};
    private static double[][] testDataLinear = { { 0, 0, 1 }, { 1, 1, 1 }, { 2, 2, 1 }, { 3, 3, 1 },
            { 0, 1, 1 }, { 1, 2, 1 }, { 2, 4.8, 1 }, { 3, 6.8, 1 },
            { 0, -1, 1 }, { 1, 0, 1 }, { 2, 1, 1 }, { 3, 2, 1 },
            { 0, -2, 1 }, { 1, -1, 1 }, { 2, 0, 1 }, { 3, 1, 1 },
            { 0, 2, 1 }, { 1, 4, 1 }, { 2, 6, 1 }, { 3, 8, 1 },
            { 0, 7.2, 1 }, { 1, 3.1, 1 }, { 2, 5.5, 1 }, { 3, 12, 1 } };
    private static int[] testOutputsLinear = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
    private static double[][] trainDataLinearMin = { { 0, 0.9, 1 }, { 5, 10.9, 1 }, { 0, 1.1, 1 }, { 5, 11.1, 1 } };
    private static int[] targetOutputsLinearMin = { 0, 0, 1, 1 };
    

    public static void main(String[] args) {

        // Data
        double[][] trainData = trainDataLinearMin;
        int[] targetOutputs = targetOutputsLinearMin;
        int range = 20;

        int activation = 0; // activation

        double[] w = initializeWeights(3, range); // Initialize weights at random
        boolean optimumFound = false;

        int counter = 0;

        for (int iteration = 0; iteration < trainData.length && !optimumFound; iteration++) {

            while (sumError > someValue && sumError <= previousValue) { // while sum_error is still getting much smaller

                sumError = 0;

                activation = calculateActivation(trainData, w, iteration);
                error = targetOutputs[iteration] - activation;
                
                System.out.println("iteration: " + iteration);
                sumError += Math.abs(error);
                System.out.println("this is sumerror: " + sumError);
                
                for (int i = 0; i < w.length; i++) {
                    d = m * error * trainData[iteration][i];
                    w[i] += d;
                    // System.out.println("d: " + d);
                }
                System.out.println(Arrays.toString(w));

                if (counter % 2 == 0) {
                    previousValue = sumError;
                }
                counter++;
            }
            if (trainData.length - 1 == iteration) {
                sumError = totalError(trainData, w, targetOutputs);
                if (sumError == 0) {
                    optimumFound = true;
                    System.out.println("Total Optimum was found!!!");
                    System.out.println("it took " + counter + " iterations");
                    System.out.println(
                            "the Wegiths are " + w[0] + " and " + w[1] + "and the weight of the Bias unit is " + w[2]);

                } else {
                    System.out.println("Optimum wasnt found yet and iteration was reset");
                    System.out.println("calculated total error is " + sumError);
                    iteration = -1;
                }

            }
            sumError = 1;
            previousValue = 2;
        }

        System.out.print("this is the totalerror with the testData: ");
        System.out.println(totalError(trainDataLinear, w, targetOutputsLinear));
    }

    public static int calculateActivation(double[][] inputs, double[] weights, int iteration) {
        double netInput = 0;
        int a;

        for (int i = 0; i < inputs[iteration].length; i++) {
            netInput += inputs[iteration][i] * weights[i];

        }
        System.out.println("this is netInput: " + netInput);
        if (netInput <= 0) { // Activation stepfunction at t = 0;
            a = 0;
        } else {
            a = 1;
        }
        return a;
    }

    public static double[] initializeWeights(int amount, int range) {
        double[] w = new double[amount]; // Weights
        double weight;
        for (int i = 0; i < amount; i++) {
            weight = Math.random() * range - (range / 2);
            w[i] = weight;
        }
        System.out.println(Arrays.toString(w));
        return w;
    }

    public static double totalError(double[][] inputs, double[] weights, int[] targetOutputs) {
        double totalSumError = 0;
        int activation = 0;
        for (int i = 0; i < inputs.length; i++) {
            activation = calculateActivation(inputs, weights, i);
            totalSumError += Math.abs(targetOutputs[i] - activation);
        }

        return totalSumError;

    }

}