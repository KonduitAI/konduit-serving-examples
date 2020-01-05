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

import com.mashape.unirest.http.Unirest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;

import java.io.File;

/**
 * Example for client test of Inference for Keras ML model using Model step .
 */
public class InferenceModelStepKerasTest {
    public static void main(String[] args) throws Exception {
        //Preparing input NDArray
        INDArray arr = Nd4j.create(new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, 1, 10);

        //Create new file to write binary input data.
        File file = new File("src/main/resources/data/test-input.zip");
        System.out.println(file.getAbsolutePath());

        BinarySerde.writeArrayToDisk(arr, file);

        String response = Unirest.post("http://localhost:3000/raw/nd4j")
                .field("input", file)
                .asString().getBody();

        System.out.print(response);


    }
}