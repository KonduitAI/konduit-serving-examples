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
import ai.konduit.serving.pipeline.step.TransformProcessStep;
import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;

/**
 * Example for Inference for DataVec ML model using PipelineStep step .
 * This illustrates only the server configuration and start server.
 */
public class InferenceModelStepDataVec {
    public static void main(String[] args) throws Exception {

        String columnName = "first";

        // Define the input schema with string column
        Schema inputSchema = new Schema.Builder()
                .addColumnString(columnName)
                .build();
        // Define the input schema with string column
        Schema outputSchema = new Schema.Builder()
                .addColumnString(columnName)
                .build();

        //  Define a transform process that operates on the defined inputs.
        TransformProcess transformProcess = new TransformProcess.Builder(inputSchema).
                appendStringColumnTransform(columnName, "two").build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .step( new TransformProcessStep(transformProcess, outputSchema)).build();

        //Print the configuration to make sure our settings correctly set.
        System.out.println(inferenceConfiguration.toJson());

        //Start inference server as per the above configurations
        DeployKonduitServing.deployInference(inferenceConfiguration, handler -> {
           if(handler.succeeded()) {
               try {
                   HttpResponse<JsonNode> response = Unirest.post(String.format("http://localhost:%s/raw/json",
                           handler.result().getServingConfig().getHttpPort()))
                           .header("Content-Type", "application/json")
                           .body("{\"first\" :\"value\"}").asJson();

                   System.out.println(response.getBody().toString());
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