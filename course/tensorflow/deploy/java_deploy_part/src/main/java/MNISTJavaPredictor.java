/*
    Tensorflow JAVA deploy script for MNIST

    Author : Sangkeun Jung (2017)
 */

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


public class MNISTJavaPredictor {

    private static byte[] readAllBytesOrExit(Path path)
    {
        // load protobuf binary format
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static float[] load_sample_data(String fn)
    {
        // Load Sample MNIST data and return as array of float

        float[] image = new float[28*28];

        try
        {
            BufferedReader buf = new BufferedReader(new FileReader(fn));
            String lineJustFetched = null;
            ArrayList<String> words = new ArrayList<>();

            String[] wordsArray;
            String[] numbersArray;

            while(true){
                lineJustFetched = buf.readLine();
                if(lineJustFetched == null){
                    break;
                }
                else{
                    wordsArray = lineJustFetched.split("\t");
                    for(String each : wordsArray){
                        if(!"".equals(each)){
                            words.add(each);
                        }
                    }
                }
            }

            int label = Integer.parseInt( words.get(0) );
            numbersArray = words.get(1).split(",");

            for(int i = 0; i < 28*28; i++)
            {
                image[i] = Float.parseFloat( numbersArray[i] );
            }

        } catch (FileNotFoundException e)
        {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }

    public static void main(String[] args) throws Exception
    {

        String model_dir   = "./data";
        String graph_fn    = "frozen_graph.pb";

        String sample_fn_1 = model_dir + "/ex_1.data";
        String sample_fn_2 = model_dir + "/ex_2.data";


        float [][] batch_images = new float[2][28*28];
        float [] s1 = load_sample_data(sample_fn_1);
        float [] s2 = load_sample_data(sample_fn_2);

        batch_images[0] = s1;
        batch_images[1] = s2;

        // check https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/Tensor
        Tensor<Float> t_batch_image = Tensor.create(batch_images, Float.class);  // t stands for tensor

        // load graph and parameters
        byte[] graph_df = readAllBytesOrExit(  Paths.get(model_dir, graph_fn) );

        try (Graph g = new Graph())
        {
            g.importGraphDef(graph_df);

            // tf.INT64 = Java.Long
            try
            (
                    Session s = new Session(g);
                 Tensor<Long> result = s.runner().feed("pl_image", t_batch_image).fetch("pred_output").run().get(0).expect(Long.class)
            )
            {
                final long[] rshape = result.shape();
                long[] pred_labels = result.copyTo(new long[2]);

                System.out.println(pred_labels[0]);
                System.out.println(pred_labels[1]);
                return;
            }
        }
    }
}