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
import ai.konduit.serving.pipeline.step.ImageLoadingStep;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;


/**
 * Example for Inference for Image ML model using Model step .
 * This illustrates only the server configuration and start server.
 */
public class InferenceModelStepImage {

    //File path for model
    public static void main(String[] args) throws IOException, InterruptedException {

        //Model config and set model type as Image
        ImageLoadingStep imageLoadingStep = ImageLoadingStep.builder()
                .imageProcessingInitialLayout("NCHW")
                .imageProcessingRequiredLayout("NHWC")
                .inputName("imgPath")
                .outputName("imageArray")
                .dimensionsConfig("default", new Long[]{240L, 320L, 3L}) // Height, width, channels
                .build();

        //ServingConfig set httpport and Input Formats
        ServingConfig servingConfig = ServingConfig.builder().httpPort(3000).
                inputDataFormat(Input.DataFormat.IMAGE).
                outputDataFormat(Output.DataFormat.ND4J).
                predictionType(Output.PredictionType.RAW).
                build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .servingConfig(servingConfig)
                .step(imageLoadingStep)
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
