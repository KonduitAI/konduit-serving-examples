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

/**
 * Example for Inference for BERT ML model using Model step .
 * This illustrates only the server configuration and start server.
 */
@NotThreadSafe
public class InferenceModelStepKeras {
    public static void main(String[] args) throws Exception {

        //File path for model
        String kerasmodelfilePath = new ClassPathResource("data/keras/embedding_lstm_tensorflow_2.h5").
                getFile().getAbsolutePath();

        //Model config and set model type as KERAS
        ModelConfig kerasModelConfig = ModelConfig.builder()
                .modelConfigType(ModelConfigType.builder().
                        modelLoadingPath(kerasmodelfilePath).
                        modelType(ModelConfig.ModelType.KERAS).build())
                .build();

        //Set the configuration of model to step
        ModelStep kerasmodelStep = ModelStep.builder()
                .modelConfig(kerasModelConfig)
                .inputName("input")
                .outputName("lstm_1")
                .build();

        //ServingConfig set httpport and Input Formats
        ServingConfig servingConfig = ServingConfig.builder().httpPort(3000).
                inputDataFormat(Input.DataFormat.ND4J).
                outputDataFormat(Output.DataFormat.JSON).
                predictionType(Output.PredictionType.RAW).
                 build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .servingConfig(servingConfig)
                .step(kerasmodelStep)
                .build();

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