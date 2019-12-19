package ai.konduit.serving.example.inference;

import ai.konduit.serving.InferenceConfiguration;
import ai.konduit.serving.config.Input;
import ai.konduit.serving.config.Output;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.configprovider.KonduitServingMain;
import ai.konduit.serving.model.ModelConfig;
import ai.konduit.serving.model.ModelConfigType;
import ai.konduit.serving.pipeline.step.ModelStep;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.io.ClassPathResource;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.nio.charset.Charset;

@NotThreadSafe
public class InferenceModelStepKeras {
    public static void main(String[] args) throws Exception {

        String kerasmodelfilePath = new ClassPathResource("data/keras/embedding_lstm_tensorflow_2.h5").
                getFile().getAbsolutePath();

        ModelConfig kerasModelConfig = ModelConfig.builder()
                .modelConfigType(ModelConfigType.builder().
                        modelLoadingPath(kerasmodelfilePath.toString()).
                        modelType(ModelConfig.ModelType.KERAS).build())
                .build();

        ModelStep kerasmodelStep = ModelStep.builder()
                .modelConfig(kerasModelConfig)
                .inputName("input")
                .outputName("lstm_1")
                .build();

        ServingConfig servingConfig = ServingConfig.builder().httpPort(3000).
                inputDataFormat(Input.DataFormat.ND4J).
                outputDataFormat(Output.DataFormat.JSON).
                 build();

        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .servingConfig(servingConfig)
                .step(kerasmodelStep)
                .build();
        System.out.println(inferenceConfiguration.toJson());

        File configFile = new File("config.json");
        FileUtils.write(configFile, inferenceConfiguration.toJson(), Charset.defaultCharset());

        KonduitServingMain.main("--configPath", configFile.getAbsolutePath());

        Thread.sleep(3600000);

        // Unirest.post("http://localhost:3000/classification/json").field( "input",arr)
      //  Unirest.post("http://localhost:3000/application/json").field( "input",arr);
       // Unirest.post("http://localhost:3000/classification/json").field( "input",tenValues);


       }
}