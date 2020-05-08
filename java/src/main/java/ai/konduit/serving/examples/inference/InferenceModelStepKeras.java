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
import ai.konduit.serving.config.ParallelInferenceConfig;
import ai.konduit.serving.deploy.DeployKonduitServing;
import ai.konduit.serving.pipeline.step.ModelStep;
import ai.konduit.serving.pipeline.step.model.KerasStep;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;

/**
 * Example for Inference for KERAS ML model using Model step .
 * This illustrates only the server configuration and start server.
 * Example for client test of Inference for Keras ML model using Model step .
 * Unirest.post will call automatically on success of server start .
 */
@NotThreadSafe
public class InferenceModelStepKeras {
    public static void main(String[] args) throws Exception {

        //File path for model
        String kerasmodelfilePath = new ClassPathResource("data/keras/embedding_lstm_tensorflow_2.h5").
                getFile().getAbsolutePath();

        //Set the configuration of model to step
        ModelStep kerasmodelStep = KerasStep.builder()
                .path(kerasmodelfilePath)
                .inputName("input")
                .outputName("lstm_1")
                .parallelInferenceConfig(ParallelInferenceConfig.builder().workers(1).build())
                .build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .step(kerasmodelStep)
                .build();

        //Print the configuration to make sure our settings correctly set.
        System.out.println(inferenceConfiguration.toJson());

        //Preparing input NDArray
        INDArray arr = Nd4j.create(new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, 1, 10);

        //Create new file to write binary input data.
        File file = new File("src/main/resources/data/test-input.zip");
        System.out.println(file.getAbsolutePath());

        if(!file.exists()) file.createNewFile();

        BinarySerde.writeArrayToDisk(arr, file);

        //Callback function  onSuccess Unirest client call.
        DeployKonduitServing.deployInference(inferenceConfiguration, handler -> {
            if(handler.succeeded()) {
                try {
                    String response = Unirest.post(String.format("http://localhost:%s/raw/nd4j",
                            handler.result().getServingConfig().getHttpPort()))
                            .field("input", file)
                            .asString().getBody();
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
