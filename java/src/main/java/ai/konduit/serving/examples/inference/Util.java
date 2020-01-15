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

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * This util used to create static methods .
 */
public class Util {
    //generating the random port for given min and max range.
    public static int randInt(int min, int max) {
        Random rand = new Random();
        int randomNum = rand.nextInt((max - min) + 1) + min;
        return randomNum;
    }

    // Generate array with random ints between 0 to upper value
    static INDArray randInt(int[] shape, int upper) {
        return Transforms.floor(Nd4j.rand(shape).mul(upper)).divi(upper);
    }

    public static File fileDownload(String fileURL){
        String fileName = "bert.zip";
        File bertTempDir = null;
        try {
            URL website = new URL(fileURL);
            ReadableByteChannel rbc = Channels.newChannel(website.openStream());
            Path tempPath = Files.createTempDirectory("Bert");
            bertTempDir = new File(String.valueOf(tempPath));
            if (!bertTempDir.exists()) {
                bertTempDir.mkdir();
            }
            bertTempDir = new File(bertTempDir + "/" + fileName);
            FileOutputStream fos = new FileOutputStream(bertTempDir);
            fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
            fos.close();
            rbc.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(bertTempDir);
        return bertTempDir;
    }

    public static File unzipFile(String zipFileDir, String searchFileName ) {
        File bertFileDir = null;
        try (ZipFile zipFile = new ZipFile(zipFileDir)) {
            Enumeration<?> enu = zipFile.entries();
            while (enu.hasMoreElements()) {
                ZipEntry zipEntry = (ZipEntry) enu.nextElement();
                if (zipEntry.getName().toLowerCase().indexOf(searchFileName) != -1) {
                    bertFileDir = new File("src/main/resources/data/bert");
                    if (!bertFileDir.exists()) {
                        bertFileDir.mkdir();
                    }
                    InputStream inputStream = zipFile.getInputStream(zipEntry);
                    bertFileDir = new File(bertFileDir + "/" + searchFileName);
                    FileOutputStream fos = new FileOutputStream(bertFileDir);
                    byte[] bytes = new byte[1024];
                    int length;
                    while ((length = inputStream.read(bytes)) >= 0) {
                        fos.write(bytes, 0, length);
                    }
                    inputStream.close();
                    fos.close();
                }
            }
            deleteTempDirectory(new File(zipFileDir));
        } catch (IOException e) {
            System.out.println("error during unzipping file :"+ e);
        }
        return bertFileDir;
    }

    private static void deleteTempDirectory(File tempDir) throws IOException {
        //get parent folder of model.jar
        File file = new File(tempDir.getParent());
        //Delete files recursively
        try{
            FileUtils.deleteDirectory(file);
            System.out.println("directory deleted");
        } catch (IOException e){
            System.out.println("error during deleting directory :"+ e);
        }

    }
}
