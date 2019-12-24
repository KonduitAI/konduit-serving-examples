package ai.konduit.serving.examples.inference;

import com.mashape.unirest.http.Unirest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;

import java.io.File;


/**
 * Example for client test of Inference for InferenceTest ML model using Model step .
 */
public class InferenceTest {
    public static void main(String[] args) throws Exception {

        //Preparing input NDArray
        INDArray arr = Nd4j.create(new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, 1, 10);

        //Create new file to write binary input data.
        File file = new File("java/src/main/resources/data/test-input.zip");
        System.out.println(file.getAbsolutePath());
        BinarySerde.writeArrayToDisk(arr, file);

        System.out.println(Unirest.post("http://localhost:3000/raw/nd4j")
                .field("input", file)
                .asJson().getBody());
    }
}