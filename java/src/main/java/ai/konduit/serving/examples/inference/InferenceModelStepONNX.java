package ai.konduit.serving.examples.inference;

import ai.konduit.serving.InferenceConfiguration;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.configprovider.KonduitServingMain;
import ai.konduit.serving.model.*;
import ai.konduit.serving.pipeline.step.PythonStep;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.io.ClassPathResource;


import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.nio.charset.Charset;
import java.util.HashMap;

@NotThreadSafe


/**
 * Example for Inference for ONNX ML model using python step .
 * This illustrates only the server configuration and start server.
 */
class InferenceModelStepONNX {
    public static void main(String[] args) throws Exception {
        //File path for model
        String working_dir = new ClassPathResource(".").getFile().getAbsolutePath();
        String python_code="";
	
	    //Set the python inputs and outputs
        HashMap<String, String> python_inputs=new HashMap<>();
        HashMap<String, String> python_outputs=new HashMap<>();
        python_inputs.put("image", "NDArray");
        python_outputs.put("boxes", "NDARRAY");

        //python configuration for input and output.
        PythonConfig  python_config= PythonConfig.builder()
                .pythonCode(python_code)
                .pythonInputs(python_inputs)
                .pythonOutputs(python_outputs)
                .pythonPath(working_dir)
                .build();

        String input_name="input1";
        //Set the configuration of python to step
        PythonStep onnx_step = PythonStep.builder()
                .inputName(input_name)
                .pythonConfig("python_config",python_config).build();

        //ServingConfig set httpport and Input Formats
        int port = Util.randInt(1000, 65535);
        ServingConfig servingConfig = ServingConfig.builder().httpPort(3000).
                build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .step(onnx_step).servingConfig(servingConfig).build();

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
