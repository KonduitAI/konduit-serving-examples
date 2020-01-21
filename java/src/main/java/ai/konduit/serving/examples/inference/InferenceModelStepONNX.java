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
import ai.konduit.serving.pipeline.step.ImageLoadingStep;
import ai.konduit.serving.pipeline.step.PythonStep;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.apache.commons.io.FileUtils;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.python.PythonVariables;
import org.nd4j.linalg.api.ndarray.INDArray;
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
        String working_dir = new ClassPathResource(".").getFile().getAbsolutePath();
        String python_code = "";
        String pythonCodePath = new ClassPathResource("scripts/onnxFacedetect.py").getFile().getAbsolutePath();


        String pythonPath = Arrays.stream(cachePackages())
                .filter(Objects::nonNull)
                .map(File::getAbsolutePath)
                .collect(Collectors.joining(File.pathSeparator));

        //python configuration for input and output.
        PythonConfig python_config = PythonConfig.builder()
                .pythonCodePath(pythonCodePath)
                .pythonInput("inputimage", PythonVariables.Type.NDARRAY.name())
                .pythonOutput("boxes", PythonVariables.Type.NDARRAY.name())
                .pythonPath("C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37\\python37.zip;" +
                        "C:\\Users\\venkat-nidrive`AppData\\Local\\Programs\\Python\\Python37\\" +
                        "DLLs;C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37\\" +
                        "lib;C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37;" +
                        "C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\" +
                        "site-packages;C:\\Users\\venkat-nidrive\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\" +
                        "pyyaml-5.2-py3.7-win-amd64.egg;c:\\projects\\konduit-new\\konduit-serving\\python")


                .build();

        //Set the configuration of python to step
        PythonStep onnx_step = new PythonStep().step(python_config);

        //ServingConfig set httpport and Input Formats
        //int port = Util.randInt(1000, 65535);
        int port = 3000;
        ServingConfig servingConfig = ServingConfig.builder()
                .httpPort(port)
                .build();

        ImageTransformProcess imageTransformProcess = new ImageTransformProcess.Builder()
                .scaleImageTransform(20.0f)
                //.resizeImageTransform(28,28)
                .build();

        ImageLoadingStep imageLoadingStep = ImageLoadingStep.builder()
                .imageProcessingInitialLayout("NCHW")
                .imageProcessingRequiredLayout("NHWC")
                .inputName("inputimage")
                .dimensionsConfig("inputimage", new Long[]{28L, 28L, 3L}) // Height, width, channels
                .imageTransformProcess("inputimage", imageTransformProcess)
                .build();
       /* //Model config and set model type as KERAS
        ImageLoadingStep imageLoadingStep = ImageLoadingStep.builder()
                .inputName("inputimage")
              //  .dimensionsConfig("default", new Long[]{240L, 320L, 2L}) // Height, width, channels
                .build();*/

        //Inference Configuration
        InferenceConfiguration inferenceConfiguration = InferenceConfiguration.builder()
                .steps(Arrays.asList(imageLoadingStep, onnx_step)).servingConfig(servingConfig).build();

        //Print the configuration to make sure our settings correctly set.
        System.out.println(inferenceConfiguration.toJson());

        File configFile = new File("config.json");
        FileUtils.write(configFile, inferenceConfiguration.toJson(), Charset.defaultCharset());

        //Preparing input images.
        // File imagePath = new File("src/main/resources/data/facedetector/1.jpg");
        File imagePath = new ClassPathResource("data/facedetector/1.jpg").getFile();
        Writable[][] output = imageLoadingStep.createRunner().transform(imagePath.toString());

        INDArray rand_image = ((NDArrayWritable) output[0][0]).get();

        //Start inference server as per the above configurations
        KonduitServingMain.builder()
                .onSuccess(() -> {
                    try {
                        String response = Unirest.post("http://localhost:3000/raw/image")
                                .field("inputimage", imagePath)
                                .asString().getBody();

                        System.out.println(response);

                        System.exit(0);
                    } catch (UnirestException e) {
                        e.printStackTrace();

                        System.exit(0);
                    }
                })
                .build()
                .runMain("--configPath", configFile.getAbsolutePath());
    }

}
