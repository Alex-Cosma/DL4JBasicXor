import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by Alex on 29/07/2016.
 */
public class XorGate {

    public static void main(String[] args) {
        INDArray input = Nd4j.zeros(4, 2);
        INDArray labels = Nd4j.zeros(4, 2);

        input.putScalar(new int[]{0, 0}, 0);
        input.putScalar(new int[]{0, 1}, 0);
        labels.putScalar(new int[]{0, 0}, 1);
        labels.putScalar(new int[]{0, 1}, 0);

        input.putScalar(new int[]{1, 0}, 0);
        input.putScalar(new int[]{1, 1}, 1);
        labels.putScalar(new int[]{1, 0}, 0);
        labels.putScalar(new int[]{1, 1}, 1);

        input.putScalar(new int[]{2, 0}, 1);
        input.putScalar(new int[]{2, 1}, 0);
        labels.putScalar(new int[]{2, 0}, 0);
        labels.putScalar(new int[]{2, 1}, 1);

        input.putScalar(new int[]{3, 0}, 1);
        input.putScalar(new int[]{3, 1}, 1);
        labels.putScalar(new int[]{3, 0}, 1);
        labels.putScalar(new int[]{3, 1}, 0);

        System.out.println(input);
        System.out.println(labels);

        DataSet ds = new DataSet(input, labels);

        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.iterations(4000);
        builder.learningRate(0.01);
        builder.seed(123);
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        builder.miniBatch(false);

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list(3);

        DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
        hiddenLayerBuilder.nIn(2);
        hiddenLayerBuilder.nOut(4);
        hiddenLayerBuilder.activation("sigmoid");
        hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

        listBuilder.layer(0, hiddenLayerBuilder.build());


        DenseLayer.Builder hiddenLayerBuilder2 = new DenseLayer.Builder();
        hiddenLayerBuilder2.nIn(4);
        hiddenLayerBuilder2.nOut(4);
        hiddenLayerBuilder2.activation("sigmoid");
        hiddenLayerBuilder2.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder2.dist(new UniformDistribution(0, 1));

        listBuilder.layer(1, hiddenLayerBuilder2.build());

        OutputLayer.Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        outputLayerBuilder.nIn(4);
        outputLayerBuilder.nOut(2);
        outputLayerBuilder.activation("sigmoid");
        outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        outputLayerBuilder.dist(new UniformDistribution(0, 1));
        listBuilder.layer(2, outputLayerBuilder.build());

        listBuilder.backprop(true);

        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.fit(ds);

        INDArray output = net.output(ds.getFeatureMatrix());

        Evaluation eval = new Evaluation(2);
        eval.eval(ds.getLabels(), output);
        System.out.println(eval.stats());

    }

}
