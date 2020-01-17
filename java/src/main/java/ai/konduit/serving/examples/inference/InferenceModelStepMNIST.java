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
import ai.konduit.serving.config.ParallelInferenceConfig;
import ai.konduit.serving.config.ServingConfig;
import ai.konduit.serving.configprovider.KonduitServingMain;
import ai.konduit.serving.configprovider.KonduitServingMainArgs;
import ai.konduit.serving.model.ModelConfig;
import ai.konduit.serving.model.ModelConfigType;
import ai.konduit.serving.model.TensorDataTypesConfig;
import ai.konduit.serving.model.TensorFlowConfig;
import ai.konduit.serving.pipeline.step.ImageLoadingStep;
import ai.konduit.serving.pipeline.step.ModelStep;
import ai.konduit.serving.verticles.inference.InferenceVerticle;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.apache.commons.io.FileUtils;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.transform.ImageTransformProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.serde.binary.BinarySerde;
import org.nd4j.tensorflow.conversion.TensorDataType;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

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

        //Set the tensor input data types
        HashMap<String, TensorDataType> input_data_types = new HashMap();
        input_data_types.put("input_layer", TensorDataType.FLOAT);

        //Model config and set model type as MNIST
        ModelConfig mnistModelConfig = TensorFlowConfig.builder()
                .tensorDataTypesConfig(TensorDataTypesConfig.builder().
                        inputDataTypes(input_data_types).build())

                .modelConfigType(ModelConfigType.builder().
                        modelLoadingPath(mnistmodelfilePath.toString()).
                        modelType(ModelConfig.ModelType.TENSORFLOW).build())
                .build();

        //Set the input and output names for model step
        List<String> input_names = new ArrayList<String>(input_data_types.keySet());
        ArrayList<String> output_names = new ArrayList<>();
        output_names.add("output_layer/Softmax");
        int port = Util.randInt(1000, 65535);

        //Set the configuration of model to step
        ModelStep bertModelStep = ModelStep.builder()
                .modelConfig(mnistModelConfig)
                .inputNames(input_names)
                .outputNames(output_names)
                .parallelInferenceConfig(ParallelInferenceConfig.builder().workers(1).build())
                .build();

        //ServingConfig set httpport and Input Formats
        ServingConfig servingConfig = ServingConfig.builder().httpPort(port).
                inputDataFormat(Input.DataFormat.NUMPY).
                //  outputDataFormat(Output.DataFormat.NUMPY).
                        predictionType(Output.PredictionType.RAW).
                        build();

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .servingConfig(servingConfig)
                .step(bertModelStep)
                .build();

        //Print the configuration to make sure our settings correctly set.
        System.out.println(inferenceConfiguration.toJson());

        File configFile = new File("config.json");
        FileUtils.write(configFile, inferenceConfiguration.toJson(), Charset.defaultCharset());


        //Start inference server as per the above configurations
        KonduitServingMainArgs args1 = KonduitServingMainArgs.builder()
                .configStoreType("file").ha(false)
                .multiThreaded(false).configPort(port)
                .verticleClassName(InferenceVerticle.class.getName())
                .configPath(configFile.getAbsolutePath())
                .build();

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
        inputString.add("images/one.png");

        for (String imagePathStr : inputString) {
            String tmpInput = new ClassPathResource(imagePathStr).getFile().getAbsolutePath();
            Writable[][] tmpOutput = imageLoadingStep.createRunner().transform(tmpInput);
            INDArray tmpImage = ((NDArrayWritable) tmpOutput[0][0]).get();
            imageArr.add(tmpImage);
        }

        System.out.println(imageArr.size());

        KonduitServingMain.builder()
                .onSuccess(() -> {
                    try {
                        for (INDArray indArray : imageArr) {
                            //Create new file to write binary input data.
                            File file = new File("src/main/resources/data/test-input.zip");
                            BinarySerde.writeArrayToDisk(indArray, file);

                            String result = Unirest.post("http://localhost:" + port + "/raw/nd4j")
                                    .field("input_layer", file)
                                    .asString().getBody();
                            System.out.println(result);
                        }
                    } catch (UnirestException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                })
                .build()
                .runMain(args1.toArgs());


    }
}