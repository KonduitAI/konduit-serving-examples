package ai.konduit.serving.examples.inference;


import com.mashape.unirest.http.Unirest;
import java.util.HashMap;

/**
 * Example for client test of Inference for DataVec ML model using Model step .
 */
public class InferenceModelStepDataVecTest {
    public static void main(String[] args) throws Exception {
        //Preparing input NDArray
       HashMap<String, String> data_input=new HashMap<>();
       data_input.put("first", "value");

        String response = Unirest.post("http://localhost:3000/raw/String")
                .field("input", data_input)
                .asString().getBody();

        System.out.print(response);
    }
}
