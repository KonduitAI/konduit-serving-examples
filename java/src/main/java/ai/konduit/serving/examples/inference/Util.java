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

import com.google.api.client.googleapis.media.MediaHttpDownloader;
import com.google.api.client.googleapis.media.MediaHttpDownloaderProgressListener;
import com.google.api.client.http.GenericUrl;
import com.google.api.client.http.javanet.NetHttpTransport;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
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

    static class CustomProgressListener implements MediaHttpDownloaderProgressListener {
        public void progressChanged(MediaHttpDownloader downloader) {
            switch (downloader.getDownloadState()) {
                case MEDIA_IN_PROGRESS:
                    System.out.println(String.format("%06.2f%% downloaded", downloader.getProgress() * 100));
                    break;
                case MEDIA_COMPLETE:
                    System.out.println("Download is complete!");
            }
        }
    }

    //generating the random port for given min and max range.
    public static int randInt(int min, int max) {
        return new Random().nextInt((max - min) + 1) + min;
    }

    // Generate array with random ints between 0 to upper value
    static INDArray randInt(int[] shape, int upper) {
        return Transforms.floor(Nd4j.rand(shape).mul(upper)).divi(upper);
    }

    public static File downloadBertModel(){
        String fileName = "bert.zip";
        File bertDownloadFile = null;
        try {
            URL website = new URL("https://deeplearning4jblob.blob.core.windows.net/testresources/bert_mrpc_frozen_v1.zip");
            Path tempPath = Files.createTempDirectory("Bert");
            bertDownloadFile = new File(tempPath.toString());

            if (!bertDownloadFile.exists()) {
                bertDownloadFile.mkdir();
            }

            bertDownloadFile = new File(bertDownloadFile, fileName);
            FileOutputStream fos = new FileOutputStream(bertDownloadFile);

            System.out.println(String.format("Bert Model file download has started. Downloading at %s. " +
                    "This may take several minutes depending on your internet speed.", bertDownloadFile.getAbsolutePath()));
            System.out.println("000.00% downloaded");

            new MediaHttpDownloader(new NetHttpTransport(), request -> {})
                    .setProgressListener(new CustomProgressListener())
                    .download(new GenericUrl(website), fos);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bertDownloadFile;
    }

    public static File unzipBertFile(String zipFileDir, String searchFileName ) throws IOException {
        String classFilePath = new ClassPathResource(".").getFile().getAbsolutePath();
        File bertFileDir = null;
        String[] targetPaths = {"src/main/resources/data/bert", classFilePath+"/data/bert"};
        try (ZipFile zipFile = new ZipFile(zipFileDir)) {
            Enumeration<?> enu = zipFile.entries();
            while (enu.hasMoreElements()) {
                ZipEntry zipEntry = (ZipEntry) enu.nextElement();
                if (zipEntry.getName().toLowerCase().contains(searchFileName)) {
                    for (String targetPath : targetPaths) {
                        bertFileDir = new File(targetPath);
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
            }
            deleteTempDirectory(new File(zipFileDir));
        } catch (IOException e) {
            System.out.println("error during unzipping file :"+ e);
        }
        return bertFileDir;
    }

    private static void deleteTempDirectory(File tempDir) throws IOException {
        //Get parent directory for zip
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
