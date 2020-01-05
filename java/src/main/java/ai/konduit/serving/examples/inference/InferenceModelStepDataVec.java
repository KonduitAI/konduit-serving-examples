package ai.konduit.serving.examples.inference;

import ai.konduit.serving.InferenceConfiguration;
import ai.konduit.serving.config.Input;
import ai.konduit.serving.config.Output;
import ai.konduit.serving.config.SchemaType;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.configprovider.KonduitServingMain;
import ai.konduit.serving.pipeline.PipelineStep;
import ai.konduit.serving.pipeline.step.TransformProcessStep;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;

import java.io.File;
import java.nio.charset.Charset;
import java.util.HashMap;


/**
 * Example for Inference for DataVec ML model using PipelineStep step .
 * This illustrates only the server configuration and start server.
 */
public class InferenceModelStepDataVec {
    public static void main(String[] args) throws Exception {

        HashMap<String, TransformProcess> transformProcess=new HashMap<>();
        transformProcess.put("first",TransformProcess.fromJson("two"));

        String column_names[]=new String[5];
        column_names[0]="first";

        SchemaType types[]=new SchemaType[5];
        types[0] = SchemaType.String;
        String schema="None";

        int port = Util.randInt(1000, 65535);

        //Set the configuration of pipeline to step
        PipelineStep transform_step= TransformProcessStep.builder().transformProcesses(transformProcess).build()
                .setInput(schema,column_names,types)
                .setOutput(schema,column_names,types);

        //ServingConfig set httpport and Input Formats
        ServingConfig servingConfig = ServingConfig.builder().httpPort(port).
                inputDataFormat(Input.DataFormat.JSON).
                outputDataFormat(Output.DataFormat.JSON).
                build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .step(transform_step).servingConfig(servingConfig).build();

        //Print the configuration to make sure our settings correctly set.
        System.out.println(inferenceConfiguration.toJson());

        File configFile = new File("config.json");
        FileUtils.write(configFile, inferenceConfiguration.toJson(), Charset.defaultCharset());

        //Start inference server as per the above configurations
        KonduitServingMain.main("--configPath", configFile.getAbsolutePath());

        //Set sleep to wait till server started before getting any request from clients.
        Thread.sleep(3600000);

    }
}
