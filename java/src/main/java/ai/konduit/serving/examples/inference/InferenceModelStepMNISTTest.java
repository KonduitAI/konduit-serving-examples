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

import ai.konduit.serving.pipeline.step.ImageLoadingStep;
import com.mashape.unirest.http.Unirest;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.transform.ImageTransformProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.ArrayList;

/**
 * Example for client test of Inference for MNIST ML model using Model step .
 */
public class InferenceModelStepMNISTTest {

    public static void main(String[] args) throws Exception {

        ImageTransformProcess imageTransformProcess = new ImageTransformProcess.Builder()
                .scaleImageTransform(20.0f)
                .resizeImageTransform(28,28)
                .build();

        ImageLoadingStep imageLoadingStep = ImageLoadingStep.builder()
                .imageProcessingInitialLayout("NCHW")
                .imageProcessingRequiredLayout("NHWC")
                .inputName("default")
                .dimensionsConfig("default", new Long[]{ 240L, 320L, 3L }) // Height, width, channels
                .imageTransformProcess("default", imageTransformProcess)
                .build();

        ArrayList<INDArray> imageArr=new ArrayList<>();
        ArrayList<String> inputString=new ArrayList<>();
        inputString.add("images/one.png");
        inputString.add("images/seven.png");
        inputString.add("images/two.png");

        for (String imagePathStr : inputString) {
            String tmpInput =  new ClassPathResource(imagePathStr).getFile().getAbsolutePath();
            Writable[][] tmpOutput = imageLoadingStep.createRunner().transform(tmpInput);
            INDArray tmpImage = ((NDArrayWritable) tmpOutput[0][0]).get();
            imageArr.add(tmpImage);
        }
        System.out.println(imageArr.size());
       // Plot plt = Plot.create();
        for (INDArray indArray : imageArr) {
            String result = Unirest.post("http://localhost:3000/raw/nd4j")
                    .field("input_layer", indArray)
                    .asString().getBody();
            System.out.println("***********************");
            System.out.println(result);
        }



//        //Writing response to output file
//        File outputImagePath = new File("src/main/resources/data/test-image-output.zip");
//        FileUtils.writeStringToFile(outputImagePath, result, Charset.defaultCharset());
//
//        System.out.println(BinarySerde.readFromDisk(outputImagePath));


    }


}