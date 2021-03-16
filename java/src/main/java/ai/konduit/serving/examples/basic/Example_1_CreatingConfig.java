package ai.konduit.serving.examples.basic;

import ai.konduit.serving.pipeline.impl.pipeline.SequencePipeline;
import ai.konduit.serving.pipeline.impl.step.logging.LoggingStep;
import ai.konduit.serving.vertx.config.InferenceConfiguration;

// Similar to what `konduit config --pipeline logging`
public class Example_1_CreatingConfig {
    public static void main(String[] args) {
        // Printing a Json configuration for logging in a sequence pipeline
        SequencePipeline sequencePipelineWithLoggingStep = SequencePipeline
                .builder()
                .add(new LoggingStep())
                .build();

        System.out.format("----------%n" +
                        "Pipeline with a Logging step output%n" +
                        "------------%n" +
                        "%s%n" +
                        "------------%n%n",
                sequencePipelineWithLoggingStep.toJson());

        // Printing an empty or default InferenceConfiguration
        InferenceConfiguration defaultInferenceConfiguration = new InferenceConfiguration();

        System.out.format("----------%n" +
                        "Default inference configuration%n" +
                        "------------%n" +
                        "%s%n" +
                        "------------%n%n",
                defaultInferenceConfiguration.toJson());

        // Printing an InferenceConfiguration with a logging step sequence pipeline
        InferenceConfiguration inferenceConfigurationWithPipeline = new InferenceConfiguration();
        inferenceConfigurationWithPipeline.pipeline(sequencePipelineWithLoggingStep);

        System.out.format("----------%n" +
                        "Inference configuration with Logging step sequence pipeline%n" +
                        "------------%n" +
                        "%s%n" +
                        "------------%n%n",
                inferenceConfigurationWithPipeline.toJson());

        // Printing InferenceConfiguration in YAML
        System.out.format("----------%n" +
                        "Inference Configuration in YAML%n" +
                        "------------%n" +
                        "%s%n" +
                        "------------%n%n",
                inferenceConfigurationWithPipeline.toYaml());
    }
}
