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
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.bytedeco.numpy.presets.numpy.cachePackages;

public class InferenceModelStepBERTTest {

    public static void main(String[] args) throws Exception {
        //Preparing input NDArray

        String pythonPath = Arrays.stream(cachePackages())
                .filter(Objects::nonNull)
                .map(File::getAbsolutePath)
                .collect(Collectors.joining(File.pathSeparator));

        String pythonCodePath = new ClassPathResource("scripts/loadnumpy.py").getFile().getAbsolutePath();
        File input0 = new ClassPathResource("data/bert/input-0.npy").getFile();
        File input1 = new ClassPathResource("data/bert/input-1.npy").getFile();
        File input4 = new ClassPathResource("data/bert/input-4.npy").getFile();

       //Create new file to write binary input data.
       //client config.
        String response = Unirest.post("http://localhost:3000/raw/numpy")
                .field("IteratorGetNext:0", input0)
                .field("IteratorGetNext:1", input1)
                .field("IteratorGetNext:4", input4)
                .asString().getBody();

        System.out.print(response);
    }
}
