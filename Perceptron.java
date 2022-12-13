import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public class Perceptron {
    private static double error = 0;
    private static double m = 0.1; // Learning Constant
    private static double range = 1; // Range
    private static double[] weights;
    private static double delta = 0; // deltaWeight
    private static double sumError = 5; // sum_error
    private static double someValue = 0.05; // some_value
    private static double previousValue = 2; // some_value
    private static double[][] trainDataAND = { { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };
    private static int[] targetOutputsAND = { 0, 0, 0, 1 };
    private static double[][] trainDataOR = { { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };
    private static int[] targetOutputsOR = { 0, 1, 1, 1 };

    private static double[][] trainDataLinear = { { 0, 0.9, 1 }, { 3, 7.2, 1 }, { 5, 10.9, 1 },
            { 0, 0, 1 }, { 1, 1, 1 }, { 2, 2, 1 }, { 3, 3, 1 }, { 0, 1.1, 1 }, { 1, 2, 1 },
            { 2, 3, 1 }, { 3, 4, 1 }, { 0, -1, 1 }, { 1, 0, 1 }, { 2, 1, 1 }, { 3, 2, 1 },
            { 0, -2, 1 }, { 1, -1, 1 }, { 2, 0, 1 }, { 3, 1, 1 }, { 3, 122, 1 },
            { 3, 6.5, 1 }, { -12, -18, 1 }, { -50, -101, 1 }, { 100, 202, 1 } };
    private static int[] targetOutputsLinear = { 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 1, 0, 1 };

    private static double[][] testDataLinear = { { 0, 0, 1 }, { 1, 1, 1 }, { 2, 2, 1 }, { 3, 3, 1 },
            { 0, 0.5, 1 }, { 1, 2, 1 }, { 2, 4.8, 1 }, { 3, 6.8, 1 },
            { 0, -1, 1 }, { 1, 0, 1 }, { 2, 1, 1 }, { 3, 2, 1 },
            { 0, -2, 1 }, { 1, -1, 1 }, { 2, 0, 1 }, { 3, 1, 1 },
            { 0, 2, 1 }, { 1, 4, 1 }, { 2, 6, 1 }, { 3, 8, 1 },
            { 0, 7.2, 1 }, { 1, 3.1, 1 }, { 2, 5.5, 1 }, { 3, 12, 1 } };
    private static int[] testOutputsLinear = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };

    private static double[][] trainDataLinearMin = { { 0, 0.9, 1 }, { 3, 7.1, 1 }, { 5, 10.9, 1 } };
    private static int[] targetOutputsLinearMin = { 0, 1, 0 };

    public static void main(String[] args) {
        System.out.println("this is the average: " + average(1, trainDataLinearMin, targetOutputsLinearMin)); // average of
        // big
        // trainDatalinear
        System.out.print(
                "this is the total error with the testData: " + totalError(testDataLinear, weights, testOutputsLinear)); // checks
        // the
        // calculated
        // weights
    }

    // returns the average of k runs and takes in trainData and targetOutput arrays
    public static double average(int k, double[][] trainData, int[] targetOutputs) {
        String str = "";
        double result = 0;
        for (int i = 0; i < k; i++) {
            int b = NN(trainData, targetOutputs);
            str += b + ", ";
            result += b;
        }
        System.out.println(str);
        return result / k;
    }

    // returns the number of adjustments the ANN took
    public static int NN(double[][] trainData, int[] targetOutputs) {

        int activation = 0; // activation

        weights = initializeWeights(3, range); // Initialize weights at random
        boolean optimumFound = false;

        int counter = 0;

        File file = new File("diagram.txt");
        try  {
            FileWriter fileWriter = new FileWriter(file);
            PrintWriter printWriter = new PrintWriter(fileWriter);

            for (int iterator = 0; !optimumFound; iterator++) { // increments iterator until optimumfound

                while (sumError > someValue && sumError <= previousValue) { // while sum_error is still getting much smaller

                    sumError = 0;
                    activation = calculateActivation(trainData, weights, iterator);
                    error = targetOutputs[iterator] - activation;
                    sumError += Math.abs(error);
                    // if the error is not 0 the weights get changed with the delta that gets
                    if (sumError != 0) { // calculated by the Hebbian learning rule and the previous value gets decreaes
                        // such that the algorithm doesnt iterate for ever
                        printWriter.println(totalError(trainData, weights, targetOutputs));
                        for (int i = 0; i < weights.length; i++) {
                            delta = m * error * trainData[iterator][i];
                            weights[i] += delta;
                        }
                        System.out.println("Adjusted weights " + Arrays.toString(weights));

                        if (counter % 2 == 0) {
                            previousValue = sumError;
                        }
                        counter++;
                    }
                }
                if (trainData.length - 1 == iterator) {
                    sumError = totalError(trainData, weights, targetOutputs);
                    if (sumError == 0) {
                        optimumFound = true;
                        System.out.println("Total Optimum was found!!!");
                        System.out.println("It took " + counter + " adjustments");
                        System.out.println(
                                "The weights are " + weights[0] + " and " + weights[1]
                                        + " and the weight of the bias unit is " + weights[2]);

                    } else {
                        System.out.println("Optimum wasn't found yet and iterator was reset");
                        System.out.println("The calculated total error was " + sumError);
                        iterator = -1;
                    }

                }
                sumError = 1;
                previousValue = 2;
            }
            return counter;
        }catch(IOException e) {
            e.printStackTrace();
            return -1;
        }
    }

    // calculates(0,1) the activation of the inputs and the weights at the current
    // vector with the step function
    public static int calculateActivation(double[][] inputs, double[] weights, int iteration) {
        double netInput = 0;
        int a;

        for (int i = 0; i < inputs[iteration].length; i++) {
            netInput += inputs[iteration][i] * weights[i];
        }
        if (netInput <= 0) { // Activation step-function at t = 0;
            a = 0;
        } else {
            a = 1;
        }
        return a;
    }

    // initialises all weights taking the range into account
    public static double[] initializeWeights(int amount, double range) {
        double[] w = new double[amount]; // Weights
        double weight;
        for (int i = 0; i < amount; i++) {
            weight = Math.random() * range - (range / 2);
            w[i] = weight;
        }
        System.out.println(Arrays.toString(w));
        return w;
    }

    // calculates total error
    public static double totalError(double[][] inputs, double[] weights, int[] targetOutputs) {
        double totalSumError = 0;
        int activation = 0;
        for (int i = 0; i < inputs.length; i++) {
            activation = calculateActivation(inputs, weights, i);
            int er = Math.abs(targetOutputs[i] - activation);
            totalSumError += er;
        }

        return totalSumError;
    }

}