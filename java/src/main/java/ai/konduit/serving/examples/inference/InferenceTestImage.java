package ai.konduit.serving.examples.inference;

import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.serde.binary.BinarySerde;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;

public class InferenceTestImage {
    public static void main(String[] args) throws IOException, UnirestException {
        File imageFile = new ClassPathResource("images/test_img.png").getFile();

        String output = Unirest.post("http://localhost:3000/RAW/IMAGE")
                .field("imgPath", imageFile)
                .asString().getBody();

        File outputImagePath = new File("java/src/main/resources/data/test-nd4j-output.zip");
        FileUtils.writeStringToFile(outputImagePath, output, Charset.defaultCharset());

        System.out.println(BinarySerde.readFromDisk(outputImagePath));
    }
}
