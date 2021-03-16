package ai.konduit.serving.examples.basic;

import ai.konduit.serving.vertx.config.InferenceConfiguration;

public class Example_1_CreatingConfig {
    public static void main(String[] args) {
        InferenceConfiguration inferenceConfiguration = new InferenceConfiguration();

        System.out.println(inferenceConfiguration.toJson());
    }
}
