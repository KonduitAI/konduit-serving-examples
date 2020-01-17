/*
 *       Copyright (c) 2019 Konduit AI.
 *
 *       This program and the accompanying materials are made available under the
 *       terms of the Apache License, Version 2.0 which is available at
 *       https://www.apache.org/licenses/LICENSE-2.0.
 *
 *       Unless required by applicable law or agreed to in writing, software
 *       distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *       WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *       License for the specific language governing permissions and limitations
 *       under the License.
 *
 *       SPDX-License-Identifier: Apache-2.0
 *
 */
package ai.konduit.serving.examples.inference;

import ai.konduit.serving.InferenceConfiguration;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.configprovider.KonduitServingMain;
import ai.konduit.serving.model.PythonConfig;
import ai.konduit.serving.pipeline.step.PythonStep;
import org.apache.commons.io.FileUtils;
import org.datavec.python.PythonVariables;
import org.nd4j.linalg.io.ClassPathResource;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.bytedeco.numpy.presets.numpy.cachePackages;

@NotThreadSafe


/**
 * Example for Inference for ONNX ML model using python step .
 * This illustrates only the server configuration and start server.
 */
class InferenceModelStepONNX {
    public static void main(String[] args) throws Exception {
        //File path for model
        String pythonCodePath = new ClassPathResource("scripts/onnxFacedetect.py").getFile().getAbsolutePath();

        String pythonPath = Arrays.stream(cachePackages())
                .filter(Objects::nonNull)
                .map(File::getAbsolutePath)
                .collect(Collectors.joining(File.pathSeparator));

        //python configuration for input and output.
        PythonConfig python_config = PythonConfig.builder()
                .pythonCode(pythonCodePath)
                .pythonInput("image", PythonVariables.Type.NDARRAY.name())
                .pythonOutput("boxes", PythonVariables.Type.NDARRAY.name())
                .pythonPath(pythonPath)
                .build();

        String input_name = "input1";

        //Set the configuration of python to step
        PythonStep onnx_step = new PythonStep().step(python_config);

        //ServingConfig set httpport and Input Formats
        //int port = Util.randInt(1000, 65535);
        int port = 3000;
        ServingConfig servingConfig = ServingConfig.builder().httpPort(port).
                //      inputDataFormat(Input.DataFormat.ND4J).
                //  outputDataFormat(Output.DataFormat.NUMPY).
                //          predictionType(Output.PredictionType.RAW).
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
