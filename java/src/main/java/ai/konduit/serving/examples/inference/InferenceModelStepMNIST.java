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
import ai.konduit.serving.pipeline.step.ImageLoadingStep;
import ai.konduit.serving.pipeline.step.ModelStep;
import ai.konduit.serving.pipeline.step.model.TensorFlowStep;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.transform.ImageTransformProcess;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.binary.BinarySerde;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Example for Inference for MNIST ML model using Model step .
 * This illustrates only the server configuration and start server.
 */
@NotThreadSafe
public class InferenceModelStepMNIST {
    public static void main(String[] args) throws Exception {

        String tensorflow_version = "2.0.0";

        //File path for model
        String mnistmodelfilePath = new ClassPathResource("data/mnist/mnist_" + tensorflow_version + ".pb").getFile().getAbsolutePath();

        //Set the configuration of model to step
        ModelStep bertModelStep = TensorFlowStep.builder()
                .path(mnistmodelfilePath)
                .inputDataType("input_layer", TensorDataType.FLOAT)
                .outputName("output_layer/Softmax")
                .parallelInferenceConfig(ParallelInferenceConfig.builder().workers(1).build())
                .build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .servingConfig(new ServingConfig())
                .step(bertModelStep)
                .build();

        //Print the configuration to make sure our settings correctly set.
        System.out.println(inferenceConfiguration.toJson());

        ImageTransformProcess imageTransformProcess = new ImageTransformProcess.Builder()
                .scaleImageTransform(20.0f)
                .resizeImageTransform(28, 28)
                .build();

        ImageLoadingStep imageLoadingStep = ImageLoadingStep.builder()
                .imageProcessingInitialLayout("NCHW")
                .imageProcessingRequiredLayout("NHWC")
                .inputName("default")
                .dimensionsConfig("default", new Long[]{240L, 320L, 3L}) // Height, width, channels
                .imageTransformProcess("default", imageTransformProcess)
                .build();

        ArrayList<INDArray> imageArr = new ArrayList<>();
        ArrayList<String> inputString = new ArrayList<>();
        inputString.add("data/facedetector/1.jpg");

        //Currently one one image tested.
        for (String imagePathStr : inputString) {
            String tmpInput = new ClassPathResource(imagePathStr).getFile().getAbsolutePath();
            Writable[][] tmpOutput = imageLoadingStep.createRunner().transform(tmpInput);
            INDArray tmpImage = ((NDArrayWritable) tmpOutput[0][0]).get();
            imageArr.add(tmpImage);
        }

        DeployKonduitServing.deployInference(inferenceConfiguration, handler -> {
           if(handler.succeeded()) {
               try {
                   for (INDArray indArray : imageArr) {

                       //Create new file to write binary input data.
                       File file = new File("src/main/resources/data/test-input.zip");
                       BinarySerde.writeArrayToDisk(indArray, file);

                       String result = Unirest.post(String.format("http://localhost:%s/raw/nd4j",
                               handler.result().getServingConfig().getHttpPort()))
                               .field("input_layer", file)
                               .asString().getBody();

                       System.out.println(result);
                   }
               } catch (UnirestException | IOException e) {
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