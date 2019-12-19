package ai.konduit.serving.examples.inference;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.Unirest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;

import java.io.File;

public class InferenceTest {
    public static void main(String[] args) throws Exception {
        INDArray arr = Nd4j.create(new float[][]{{1, 0, 5, 10}, {100, 55, 555, 1000}});

        File file = new File("C:\\Input\\arr.zip");
        BinarySerde.writeArrayToDisk(arr, file);

        HttpResponse<JsonNode> jsonResponse =
                Unirest.post("http://localhost:3000/raw/nd4j").
                        header("content-type", "multipart/form-data").field("input", file.getAbsolutePath()).asJson();

        System.out.print(jsonResponse);
    }
}