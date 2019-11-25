package ai.konduit.serving.examples.utils;

/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.SimpleCNN;

import java.io.File;

/**
Initialize and save a SimpleCNN model from the Deeplearning4j model zoo
 */
public class SaveSimpleCNN {
    private static int nClasses = 5;
    private static boolean saveUpdater = false;

    public static void main(String[] args) throws Exception {
        ZooModel zooModel = SimpleCNN.builder()
                .numClasses(nClasses)
                .inputShape(new int[]{3, 224, 224})
                .build();
        MultiLayerConfiguration conf = ((SimpleCNN) zooModel).conf();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        System.out.println(net.summary());

        // Save the model: Where to save the network.
        // Note: the file is in .zip format - can be opened externally
        File locationToSave = new File("SimpleCNN.zip");
        // Updater: Save this if you want to train your network more in the future
        net.save(locationToSave, saveUpdater);
    }
}
