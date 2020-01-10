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
import ai.konduit.serving.config.Input;
import ai.konduit.serving.config.ParallelInferenceConfig;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.configprovider.KonduitServingMain;
import ai.konduit.serving.model.ModelConfig;
import ai.konduit.serving.model.ModelConfigType;
import ai.konduit.serving.model.TensorDataTypesConfig;
import ai.konduit.serving.pipeline.step.ModelStep;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.tensorflow.conversion.TensorDataType;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Example for Inference for DL4J ML model using Model step .
 * This illustrates only the server configuration and start server.
 */
@NotThreadSafe
public class InferenceModelStepDL4J {
    public static void main(String[] args) throws Exception {

        //File path for model
        String dl4jmodelfilePath = new ClassPathResource("data/multilayernetwork/SimpleCNN.zip").
                getFile().getAbsolutePath();

        //Set the tensor input data types
        Map<String, TensorDataType> input_data_types = new HashMap<>();
        input_data_types.put("image_array", TensorDataType.FLOAT);

        //Set the input and output names for model step
        List<String> input_names = new ArrayList<String>(input_data_types.keySet());
        List<String> output_names = new ArrayList<>();
        output_names.add("output");
        int port = Util.randInt(1000, 65535);

        //Model config and set model type as DL4J
        ModelConfig dl4jModelConfig = ModelConfig.builder()
                .tensorDataTypesConfig(TensorDataTypesConfig.builder().
                        inputDataTypes(input_data_types).build())
                .modelConfigType(ModelConfigType.builder().
                        modelLoadingPath(dl4jmodelfilePath.toString()).
                        modelType(ModelConfig.ModelType.MULTI_LAYER_NETWORK).build())
                .build();

        //Set the configuration of model to step
        ModelStep dl4jModelStep = ModelStep.builder()
                .modelConfig(dl4jModelConfig)
                .inputNames(input_names)
                .outputNames(output_names)
                .parallelInferenceConfig(ParallelInferenceConfig.builder().workers(1).build())
                .build();

        //ServingConfig set httpport and Input Formats
        ServingConfig servingConfig = ServingConfig.builder().httpPort(3000).
                inputDataFormat(Input.DataFormat.ND4J).
                // outputDataFormat(Output.DataFormat.ND4J).
                        build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .servingConfig(servingConfig)
                .step(dl4jModelStep)
                .build();

        //Print the configuration to make sure our settings correctly set.
        System.out.println(dl4jModelConfig);
        System.out.println(inferenceConfiguration.toJson());

        File configFile = new File("config.json");
        FileUtils.write(configFile, inferenceConfiguration.toJson(), Charset.defaultCharset());

        //Start inference server as per the above configurations
        KonduitServingMain.main("--configPath", configFile.getAbsolutePath());

        //Set sleep to wait till server started before getting any request from clients.
        Thread.sleep(3600000);

    }
}
