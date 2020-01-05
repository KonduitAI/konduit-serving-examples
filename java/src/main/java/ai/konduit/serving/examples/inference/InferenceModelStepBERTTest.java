package ai.konduit.serving.examples.inference;

import ai.konduit.serving.model.PythonConfig;
import ai.konduit.serving.pipeline.step.PythonStep;
import com.mashape.unirest.http.Unirest;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.python.PythonVariables;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
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
        String input0 =  new ClassPathResource("data/bert/input-0.npy").getFile().getAbsolutePath();
        String input1 =  new ClassPathResource("data/bert/input-1.npy").getFile().getAbsolutePath();
        String input4 =  new ClassPathResource("data/bert/input-4.npy").getFile().getAbsolutePath();

        PythonConfig pythonConfig = PythonConfig.builder()
                .pythonPath(pythonPath) // If not null, this python path will be used.
                .pythonCodePath(pythonCodePath)
                .pythonInput("x", PythonVariables.Type.STR.name())
                .pythonOutput("y", PythonVariables.Type.NDARRAY.name())
                .build();

        PythonStep pythonPipelineStep = new PythonStep().step(pythonConfig);

        Writable[][] output0 = pythonPipelineStep.createRunner().transform(input0);
        Writable[][] output1 = pythonPipelineStep.createRunner().transform(input1);
        Writable[][] output4 = pythonPipelineStep.createRunner().transform(input4);

        INDArray image0 = ((NDArrayWritable) output0[0][0]).get();
        INDArray image1 = ((NDArrayWritable) output1[0][0]).get();
        INDArray image4 = ((NDArrayWritable) output4[0][0]).get();

        HashMap<String, INDArray> data_input=new HashMap<>();
        data_input.put("IteratorGetNext:0",image0);
        data_input.put("IteratorGetNext:1",image1);
        data_input.put("IteratorGetNext:4",image4);

        //System.out.println(data_input);

        //client config.
        String response = Unirest.post("http://localhost:3000/raw/nd4j")
                .field("input", data_input)
                .asString().getBody();

        System.out.print(response);
    }
}
