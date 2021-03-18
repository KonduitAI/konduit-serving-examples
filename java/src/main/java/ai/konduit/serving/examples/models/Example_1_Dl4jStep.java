package ai.konduit.serving.examples.models;

import ai.konduit.serving.examples.utils.Train;
import ai.konduit.serving.models.deeplearning4j.step.DL4JStep;
import ai.konduit.serving.pipeline.impl.pipeline.SequencePipeline;
import ai.konduit.serving.vertx.api.DeployKonduitServing;
import ai.konduit.serving.vertx.config.InferenceConfiguration;
import ai.konduit.serving.vertx.config.InferenceDeploymentResult;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import io.vertx.core.DeploymentOptions;
import io.vertx.core.VertxOptions;
import org.json.JSONObject;

public class Example_1_Dl4jStep {
    public static void main(String[] args) throws Exception {
        InferenceConfiguration inferenceConfiguration = new InferenceConfiguration();
        inferenceConfiguration.pipeline(SequencePipeline.builder()
                .add(new DL4JStep()
                        .modelUri(Train.dl4jIrisModel())
                        .inputNames("layer0")
                        .outputNames("layer0"))
                .build()
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
                            String result = Unirest.post(String.format("http://localhost:%s/predict", runnningPort))
                                    .header("Content-Type", "application/json")
                                    .header("Accept", "application/json")
                                    .body(new JSONObject().put("layer0", "input_value"))
                                    .asString().getBody();

                            System.out.format("Result from server : %s%n", result);

                            System.exit(0);
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
