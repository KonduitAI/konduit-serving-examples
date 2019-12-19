package inference;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.Unirest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.serde.binary.BinarySerde;

import java.io.File;

public class InferenceTest {
    public static void main(String[] args) throws Exception {
        INDArray arr = Nd4j.create(new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, 1, 10);

        File file = new File("java/src/main/resources/data/test-input.zip");
        System.out.println(file.getAbsolutePath());
        BinarySerde.writeArrayToDisk(arr, file);

        System.out.println(Unirest.post("http://localhost:3000/raw/nd4j")
                .field("input", file)
                .asString().getBody());
    }
}