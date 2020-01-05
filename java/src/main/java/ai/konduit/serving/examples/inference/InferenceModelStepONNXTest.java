/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2019 Konduit AI.
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *  Unless required by applicable law or agreed to in writing, software
 *  *  *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package ai.konduit.serving.examples.inference;

import ai.konduit.serving.pipeline.step.ImageLoadingStep;
import com.mashape.unirest.http.Unirest;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.transform.ImageTransformProcess;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

/**
 * Example for client test of Inference for ONNX ML model using Model step .
 */

public class InferenceModelStepONNXTest {
    public static void main(String[] args) throws Exception {

        ImageTransformProcess imageTransformProcess = new ImageTransformProcess.Builder()
                .scaleImageTransform(20.0f)
                //.resizeImageTransform(28,28)
                .build();

        ImageLoadingStep imageLoadingStep = ImageLoadingStep.builder()
                .imageProcessingInitialLayout("NCHW")
                .imageProcessingRequiredLayout("NHWC")
                .inputName("default")
                .dimensionsConfig("default", new Long[]{ 240L, 320L, 3L }) // Height, width, channels
                .imageTransformProcess("default", imageTransformProcess)
                .build();
        //Preparing input images.
        File imagePath =  new File("konduit-serving-examples/src/main/resources/data/facedetector/1.jpg");

        Writable[][] output = imageLoadingStep.createRunner().transform(imagePath.toString());

        INDArray rand_image = ((NDArrayWritable) output[0][0]).get();

        System.out.println(rand_image);

        String response = Unirest.post("http://localhost:3000/raw/nd4j")
                .field("input", rand_image)
                .asString().getBody();

        System.out.println(response);


    }
}
