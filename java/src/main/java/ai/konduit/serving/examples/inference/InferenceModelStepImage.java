package ai.konduit.serving.examples.inference;

import ai.konduit.serving.InferenceConfiguration;
import ai.konduit.serving.config.Input;
import ai.konduit.serving.config.Output;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.configprovider.KonduitServingMain;
import ai.konduit.serving.pipeline.step.ImageLoadingStep;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;

public class InferenceModelStepImage {
    public static void main(String[] args) throws IOException, InterruptedException {
        //Model config and set model type as KERAS
        ImageLoadingStep imageLoadingStep = ImageLoadingStep.builder()
                .imageProcessingInitialLayout("NCHW")
                .imageProcessingRequiredLayout("NHWC")
                .inputName("imgPath")
                .outputName("imageArray")
                .dimensionsConfig("default", new Long[]{ 240L, 320L, 3L }) // Height, width, channels
                .build();

        //ServingConfig set httpport and Input Formats
        ServingConfig servingConfig = ServingConfig.builder().httpPort(3000).
                inputDataFormat(Input.DataFormat.IMAGE).
                outputDataFormat(Output.DataFormat.ND4J).
                predictionType(Output.PredictionType.RAW).
                build();

        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .servingConfig(servingConfig)
                .step(imageLoadingStep)
                .build();
        System.out.println(inferenceConfiguration.toJson());

        File configFile = new File("config.json");
        FileUtils.write(configFile, inferenceConfiguration.toJson(), Charset.defaultCharset());

        KonduitServingMain.main("--configPath", configFile.getAbsolutePath());

        Thread.sleep(3600000);
    }
}
