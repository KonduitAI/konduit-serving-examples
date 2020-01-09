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

import com.mashape.unirest.http.Unirest;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Example for client test of Inference for DL4J ML model using Model step .
 */
public class InferenceModelStepDL4JTest {
    public static void main(String[] args) throws Exception {

        //Create random array between 0-244
        INDArray rand_image = Util.randInt(new int[]{1, 3, 244, 244}, 255);

        String response = Unirest.post("http://localhost:3000/raw/nd4j")
                .field("image_array", rand_image)
                .asString().getBody();

        System.out.println(response);

    }
}
