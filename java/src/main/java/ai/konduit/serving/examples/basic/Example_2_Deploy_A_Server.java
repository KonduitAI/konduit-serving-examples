package ai.konduit.serving.examples.basic;

import ai.konduit.serving.pipeline.impl.pipeline.SequencePipeline;
import ai.konduit.serving.pipeline.impl.step.logging.LoggingStep;
import ai.konduit.serving.vertx.api.DeployKonduitServing;
import ai.konduit.serving.vertx.config.InferenceConfiguration;
import ai.konduit.serving.vertx.config.InferenceDeploymentResult;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import io.vertx.core.DeploymentOptions;
import io.vertx.core.VertxOptions;
import org.json.JSONObject;

// Similar to what we can achieve with `konduit serve` command
public class Example_2_Deploy_A_Server {
    public static void main(String[] args) {
        // Creating an inference configuration with logging step
        InferenceConfiguration inferenceConfiguration = new InferenceConfiguration();
        inferenceConfiguration.pipeline(
                SequencePipeline
                        .builder()
                        .add(new LoggingStep().log(LoggingStep.Log.KEYS_AND_VALUES))
                        .build()
        );

        DeployKonduitServing.deploy(
                new VertxOptions(), // Default vertx options
                new DeploymentOptions(), // Default deployment options
                inferenceConfiguration, // Inference configuration with logging step
                handler -> { // this block will be called when server finishes the deployment
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
                                    .body(new JSONObject().put("input_key", "input_value"))
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
