package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.support.v7.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

public class ObjectDetectionJar {
    private Interpreter interpreter = null;
    private String TFLITE_MODEL_NAME = "best_float16.tflite";

    public void createInterpreter(Context context) {
        Interpreter.Options tfLiteOptions = new Interpreter.Options(); //can be configure to use GPUDelegat

        try {
            interpreter = new Interpreter(FileUtil.loadMappedFile(context, TFLITE_MODEL_NAME), tfLiteOptions);
        } catch (IOException callback) {
            System.out.println(callback);
        }
    }

    private ByteBuffer getInputImage(int width, int height) {
        ByteBuffer inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4); // input image will be required input shape of tflite model
        inputImage.order(ByteOrder.nativeOrder());
        inputImage.rewind();
        return inputImage;
    }


    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmapIn, int width, int height){
        Bitmap bitmap = Bitmap.createScaledBitmap(bitmapIn, width, height, false); // convert bitmap into required size
        // these value can be different for each channel if they are not then you may have single value instead of an array
        float[] mean = new float[]{127.5f, 127.5f, 127.5f};
        float[] standard = new float[]{127.5f, 127.5f, 127.5f};

        ByteBuffer inputImage = getInputImage(width, height);
        int[] intValues = new int[width * height];
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);
        for (int y=0; y<width; y++) {
            for (int x=0; x<height; x++) {
                int px = bitmap.getPixel(x, y);
                // Get channel values from the pixel value.
                int r = Color.red(px);
                int g = Color.green(px);
                int b = Color.blue(px);
                // Normalize channel values to [-1.0, 1.0]. This requirement depends on the model.
                // For example, some models might require values to be normalized to the range
                // [0.0, 1.0] instead.
                float rf = (r - mean[0]) / standard[0];
                float gf = (g - mean[0]) / standard[0];
                float bf = (b - mean[0]) / standard[0];
                //putting in BRG order because this model demands input in this order
                inputImage.putFloat(bf);
                inputImage.putFloat(rf);
                inputImage.putFloat(gf);
            }
        }
        return inputImage;
    }

    public ArrayList<ArrayList<ArrayList<Float>>> runInference(Bitmap bitmap) {
        //float[][][][] outputArr = new float[1][480][480][3];
        ArrayList<ArrayList<ArrayList<Float>>> outputArr = new ArrayList<>();//new Array   1,   15, 4725;
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap, 480, 480);
        interpreter.run(byteBuffer, outputArr);
        interpreter.close(); // close interpreter
        return outputArr;
    }
}
