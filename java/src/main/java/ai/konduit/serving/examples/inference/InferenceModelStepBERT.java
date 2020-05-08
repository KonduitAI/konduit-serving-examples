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
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.deploy.DeployKonduitServing;
import ai.konduit.serving.model.TensorDataType;
import ai.konduit.serving.pipeline.step.ModelStep;
import ai.konduit.serving.pipeline.step.model.TensorFlowStep;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.nd4j.common.io.ClassPathResource;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.util.ArrayList;


/**
 * Example for Inference for BERT ML model using Model step .
 * This illustrates only the server configuration and start server.
 */
@NotThreadSafe
public class InferenceModelStepBERT {
    public static void main(String[] args) throws Exception {

        //File path for model
        String bertDataFolder = new ClassPathResource("data/bert").getFile().getAbsolutePath();

        String bertFileName = "bert_mrpc_frozen.pb";
        File bertModelFile = new File(bertDataFolder, bertFileName);

        // If bert_mrpc_frozen file doesn't exist, download it and unzip it to target folder.
        // This might take several minutes depending on the internet speed.
        if (!bertModelFile.exists()) {
            File bertDownloadedZipFile = Util.downloadBertModel();
            Util.unzipBertFile(bertDownloadedZipFile.toString(), bertFileName);
        }

        String[] inputNames = new String[] {
                "IteratorGetNext:0",
                "IteratorGetNext:1",
                "IteratorGetNext:4"
        };

        ArrayList<String> output_names = new ArrayList<>();
        output_names.add("loss/Softmax");

        //Set the configuration of model to step
        ModelStep bertModelStep = TensorFlowStep.builder()
                .inputDataType(inputNames[0], TensorDataType.INT32)
                .inputDataType(inputNames[1], TensorDataType.INT32)
                .inputDataType(inputNames[2], TensorDataType.INT32)
                .path(bertModelFile.getAbsolutePath())
                .outputNames(output_names)
                .parallelInferenceConfig(ParallelInferenceConfig.builder().workers(1).build())
                .build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .step(bertModelStep)
                .build();

        //Print the configuration to make sure our settings correctly set.
        System.out.println(inferenceConfiguration.toJson());

        File input0 = new ClassPathResource("data/bert/input-0.npy").getFile();
        File input1 = new ClassPathResource("data/bert/input-1.npy").getFile();
        File input4 = new ClassPathResource("data/bert/input-4.npy").getFile();

        DeployKonduitServing.deployInference(inferenceConfiguration, handler -> {
            if(handler.succeeded()) {
                try {
                    //client config.
                    String response = Unirest.post(String.format("http://localhost:%s/raw/numpy",
                            handler.result().getServingConfig().getHttpPort()))
                            .field(inputNames[0], input0)
                            .field(inputNames[1], input1)
                            .field(inputNames[2], input4)
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
