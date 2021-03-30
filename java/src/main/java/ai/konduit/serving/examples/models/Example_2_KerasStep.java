package ai.konduit.serving.examples.models;

import ai.konduit.serving.data.image.convert.ImageToNDArrayConfig;
import ai.konduit.serving.data.image.convert.config.NDChannelLayout;
import ai.konduit.serving.data.image.convert.config.NDFormat;
import ai.konduit.serving.data.image.step.ndarray.ImageToNDArrayStep;
import ai.konduit.serving.examples.utils.Train;
import ai.konduit.serving.models.deeplearning4j.step.keras.KerasStep;
import ai.konduit.serving.pipeline.impl.pipeline.SequencePipeline;
import ai.konduit.serving.vertx.api.DeployKonduitServing;
import ai.konduit.serving.vertx.config.InferenceConfiguration;
import ai.konduit.serving.vertx.config.InferenceDeploymentResult;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import io.vertx.core.DeploymentOptions;
import io.vertx.core.VertxOptions;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

public class Example_2_KerasStep {
    public static void main(String[] args) throws IOException {
        Train.ModelTrainResult modelTrainResult = Train.kerasMnistModel();

        InferenceConfiguration inferenceConfiguration = new InferenceConfiguration();
        inferenceConfiguration.pipeline(SequencePipeline.builder()
                .add(new ImageToNDArrayStep()
                        .config(new ImageToNDArrayConfig()
                                .width(28)
                                .height(28)
                                .includeMinibatchDim(true)
                                .channelLayout(NDChannelLayout.GRAYSCALE)
                                .format(NDFormat.CHANNELS_LAST)
                        )
                        .keys("image")
                        .outputNames("input_layer"))
                .add(new KerasStep()
                        .modelUri(modelTrainResult.modelPath())
                        .inputNames(modelTrainResult.inputNames())
                        .outputNames(modelTrainResult.outputNames())
                ).build()
        );

        DeployKonduitServing.deploy(new VertxOptions(), new DeploymentOptions(),
                inferenceConfiguration,
                handler -> {
                    if (handler.succeeded()) { // If the server is sucessfully running
                        // Getting the result of the deployment
                        InferenceDeploymentResult inferenceDeploymentResult = handler.result();
                        int runnningPort = inferenceDeploymentResult.getActualPort();
                        String deploymentId = inferenceDeploymentResult.getDeploymentId();

                        System.out.format("The server is running on port %s with deployment id of %s%n",
                                runnningPort, deploymentId);

                        try {
                            String result;
                            try {
                                result = Unirest.post(String.format("http://localhost:%s/predict", runnningPort))
                                        .header("Accept", "application/json")
                                        .field("image", new ClassPathResource("inputs/mnist-image-2.jpg").getFile(), "image/jpg")
                                        .asString().getBody();

                                System.out.format("Result from server : %s%n", result);

                                System.exit(0);
                            } catch (IOException e) {
                                e.printStackTrace();
                                System.exit(1);
                            }
                        } catch (UnirestException e) {
                            e.printStackTrace();
                            System.exit(1);
                        }
                    } else { // If the server failed to run
                        System.out.println(handler.cause().getMessage());
                        System.exit(1);
                    }
                });
    }
}
