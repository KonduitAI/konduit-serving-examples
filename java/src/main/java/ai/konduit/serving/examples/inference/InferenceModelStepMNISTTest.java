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
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.serde.binary.BinarySerde;

import java.io.File;
import java.nio.charset.Charset;

public class InferenceModelStepMNISTTest {
    public static void main(String[] args) throws Exception {
        String str = new ClassPathResource("images/COCO_train2014_000000000009.jpg").getFile().getAbsolutePath();

        File imageFile= new File (str);
        String output = Unirest.post("http://localhost:3000/RAW/IMAGE")
                .field("input_layer", imageFile)
                .asString().getBody();

        File outputImagePath = new File("src/main/resources/data/test-image-output.zip");
        FileUtils.writeStringToFile(outputImagePath, output, Charset.defaultCharset());

        System.out.println(BinarySerde.readFromDisk(outputImagePath));


    }
}