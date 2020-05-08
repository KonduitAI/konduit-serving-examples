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
import ai.konduit.serving.deploy.DeployKonduitServing;
import ai.konduit.serving.model.TensorDataType;
import ai.konduit.serving.pipeline.step.model.Dl4jStep;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.binary.BinarySerde;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;

/**
 * Example for Inference for DL4J ML model using Model step .
 * This illustrates only the server configuration and start server.
 */
@NotThreadSafe
public class InferenceModelStepDL4J {
    public static void main(String[] args) throws Exception {

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .servingConfig(new ServingConfig())
                .step(Dl4jStep.builder()
                        .path(new ClassPathResource("data/multilayernetwork/SimpleCNN.zip").getFile().getAbsolutePath())
                        .inputDataType("image_array",  TensorDataType.FLOAT)
                        .outputName("output")
                        .build())
                .build();

        //Print the configuration to make sure our settings correctly set.
        System.out.println(inferenceConfiguration.toJson());

        //Start inference server as per the above configurations

        INDArray randImage = Util.randInt(new int[]{1, 3, 244, 244}, 255);

        File file = new File("src/main/resources/data/test-dl4j.zip");

        if(!file.exists()) file.createNewFile();

        BinarySerde.writeArrayToDisk(randImage, file);
        System.out.println(randImage);

        DeployKonduitServing.deployInference(inferenceConfiguration, handler -> {
            if(handler.succeeded()) {
                try {
                    String response = Unirest.post(String.format("http://localhost:%s/raw/nd4j",
                            handler.result().getServingConfig().getHttpPort()))
                            .field("image_array", file).asString().getBody();
                    System.out.print(response);
                } catch (UnirestException e) {
                    e.printStackTrace();
                }

                System.exit(0);
            } else {
                handler.cause().printStackTrace();
                System.exit(1);
            }
        });
    }
}
