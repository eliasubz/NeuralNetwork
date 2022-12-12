import java.util.Arrays;

public class Perceptron {
    private static double error = 0;
    private static double learningConstant = 0.1; // Learning Constant
    private static double range = 1; // Range
    private static double[] weights;
    private static double d = 0; // deltaWeight
    private static double sumError = 1; // sum_error
    private static double someValue = 0.05; // some_value
    private static double previousValue = 2; // some_value
    private static double[][] trainDataAND = { { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };
    private static int[] targetOutputsAND = { 0, 0, 0, 1 };
    private static double[][] trainDataOR = { { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } };
    private static int[] targetOutputsOR = { 0, 1, 1, 1 };
    private static double[][] trainDataLinear = { { 0, 0.9, 1 }, { 3, 7.1, 1 }, { 5, 10.9, 1 },{ 0, 0, 1 }, { 1, 1, 1 }, { 2, 2, 1 }, { 3, 3, 1 }, { 0, 1.1, 1 },
            { 1, 2, 1 }, { 2, 3, 1 }, { 3, 4, 1 }, { 0, -1, 1 }, { 1, 0, 1 }, { 2, 1, 1 }, { 3, 2, 1 },
            { 0, -2, 1 }, { 1, -1, 1 }, { 2, 0, 1 }, { 3, 1, 1 }, { 3, 122, 1 }, { 3, 6.5, 1 }, { -12, -18, 1 },
            { -50, -101, 1 }, { 100, 202, 1 } };
    private static int[] targetOutputsLinear = { 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1 };

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
        System.out.println("this is the average: " + avarage(20,trainDataLinearMin,targetOutputsLinearMin));
        // System.out.println("this is the average: " + avarage(1,trainDataLinear,targetOutputsLinear));

        System.out.print("this is the totalerror with the testData: " + totalError(testDataLinear, weights, testOutputsLinear));
    }

    public static double avarage(int k, double[][] trainData, int[] targetOutputs) {
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

    public static int NN(double[][] trainData, int[] targetOutputs) {

        int activation = 0; // activation

        weights = initializeWeights(3, range); // Initialize weights at random
        boolean optimumFound = false;

        int counter = 0;

        for (int iterator = 0; !optimumFound; iterator++) {

            while (sumError > someValue && sumError <= previousValue) { // while sum_error is still getting much smaller

                sumError = 0;

                activation = calculateActivation(trainData, weights, iterator);
                error = targetOutputs[iterator] - activation;
                // System.out.println("this will indicate the direction "+error);
                // System.out.println("iteration: " + iteration);
                sumError += Math.abs(error);
                // System.out.println("this is sumerror: " + sumError);

                if (sumError != 0) {
                    for (int i = 0; i < weights.length; i++) {
                        d = learningConstant * error * trainData[iterator][i];
                        weights[i] += d;
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

        // System.out.print("this is the totalerror with the testData: ");
        // System.out.println(totalError(trainDataLinear, w, targetOutputsLinear));

        // System.out.println("Counter: " + counter);
        return counter;
    }

    public static int calculateActivation(double[][] inputs, double[] weights, int iteration) {
        double netInput = 0;
        int a;

        for (int i = 0; i < inputs[iteration].length; i++) {
            netInput += inputs[iteration][i] * weights[i];
        }
        // System.out.println("this is netInput: " + netInput);
        if (netInput <= 0) { // Activation stepfunction at t = 0;
            a = 0;
        } else {
            a = 1;
        }
        return a;
    }

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

    public static double totalError(double[][] inputs, double[] weights, int[] targetOutputs) {
        double totalSumError = 0;
        int activation = 0;
        for (int i = 0; i < inputs.length; i++) {
            activation = calculateActivation(inputs, weights, i);
            int er = Math.abs(targetOutputs[i] - activation);
            if (er > 0){
                System.out.println("position: " + i);
                System.out.println("activation: " + activation);
                System.out.println("inputs: " + Arrays.toString(inputs[i]));
            }
            totalSumError += er;
        }

        return totalSumError;
    }

}