package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.DataType;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class RobotVisionProcessor {

    private static final String TAG = "RobotVisionProcessor";

    private Interpreter interpreter;
    private int modelInputWidth;
    private int modelInputHeight;
    private int modelInputChannels;
    private boolean isInputFloat;

    private List<String> labels;
    private Context context;

    // --- ค่าคงที่ที่แก้ไขตามโครงสร้าง Output ของโมเดล [1, 15, 4725] ---
    // NUM_PREDICTION_SLOTS คือมิติที่ 3 ของ output tensor (4725)
    private static final int NUM_PREDICTION_SLOTS = 4725;

    // Offsets/Index ในมิติที่ 2 (จาก 15 มิติ) ของ output tensor
    // !!! การเปลี่ยนแปลง: ตั้งชื่อให้ชัดเจนว่าเป็น x_center, y_center, width, height !!!
    private static final int X_CENTER_COORD_ROW_INDEX = 0; // แถวสำหรับ x_center
    private static final int Y_CENTER_COORD_ROW_INDEX = 1; // แถวสำหรับ y_center
    private static final int WIDTH_COORD_ROW_INDEX = 2;    // แถวสำหรับ width
    private static final int HEIGHT_COORD_ROW_INDEX = 3;   // แถวสำหรับ height
    private static final int FIRST_CLASS_PROB_ROW_INDEX = 4; // Class probabilities เริ่มจากแถวที่ 4 ไปจนถึง 4 + NUM_CLASSES - 1 (คือ 14)

    public static final float CONFIDENCE_THRESHOLD = 0.5f;
    private static final float IOU_THRESHOLD = 0.45f;
    private static final int NUM_CLASSES = 11; // จำนวนคลาสทั้งหมด

    // Class for storing individual detection results
    public static class DetectionResult {
        public float xmin, ymin, xmax, ymax; // Normalized coordinates [0, 1]
        public float confidence;
        public int classId;
        public float classProb;
        public String label;

        public DetectionResult(float xmin, float ymin, float xmax, float ymax,
                               float confidence, int classId, float classProb, String label) {
            this.xmin = xmin;
            this.ymin = ymin;
            this.xmax = xmax;
            this.ymax = ymax;
            this.confidence = confidence;
            this.classId = classId;
            this.classProb = classProb;
            this.label = label;
        }

        // Method to calculate Intersection over Union (IoU)
        public float calculateIoU(DetectionResult other) {
            float intersectionXmin = Math.max(xmin, other.xmin);
            float intersectionYmin = Math.max(ymin, other.ymin);
            float intersectionXmax = Math.min(xmax, other.xmax);
            float intersectionYmax = Math.min(ymax, other.ymax);

            float intersectionWidth = Math.max(0, intersectionXmax - intersectionXmin);
            float intersectionHeight = Math.max(0, intersectionYmax - intersectionYmin);

            float intersectionArea = intersectionWidth * intersectionHeight;

            float thisArea = (xmax - xmin) * (ymax - ymin);
            float otherArea = (other.xmax - other.xmin) * (other.ymax - other.ymin);

            float unionArea = thisArea + otherArea - intersectionArea;

            if (unionArea == 0) {
                return 0;
            }
            return intersectionArea / unionArea;
        }

        @Override
        public String toString() {
            return String.format("%s (Conf: %.2f, Prob: %.2f) BBox: [%.2f,%.2f,%.2f,%.2f]",
                    label, confidence, classProb, xmin, ymin, xmax, ymax);
        }
    }

    public RobotVisionProcessor(Context context, String modelAssetName, String labelsAssetName,
                                int inputWidth, int inputHeight, int inputChannels, boolean isInputFloat) throws IOException {
        this.context = context;
        this.modelInputWidth = inputWidth;
        this.modelInputHeight = inputHeight;
        this.modelInputChannels = inputChannels;
        this.isInputFloat = isInputFloat;

        try {
            MappedByteBuffer modelBuffer = loadModelFileFromAssets(context, modelAssetName);
            Interpreter.Options options = new Interpreter.Options();
            interpreter = new Interpreter(modelBuffer, options);
            Log.d(TAG, "LiteRT model loaded successfully from assets: " + modelAssetName);

            int[] outputShape = interpreter.getOutputTensor(0).shape();
            Log.d(TAG, "Model output tensor shape: " + Arrays.toString(outputShape));
            // ตรวจสอบ Output Shape ให้ตรงกับที่คาดไว้ [1, 15, 4725]
            if (outputShape.length != 3 || outputShape[0] != 1 || outputShape[1] != (4 + NUM_CLASSES) || outputShape[2] != NUM_PREDICTION_SLOTS) {
                Log.e(TAG, "Mismatch in expected model output tensor shape! Expected [1, " + (4 + NUM_CLASSES) + ", " + NUM_PREDICTION_SLOTS + "] but got " + Arrays.toString(outputShape) + ". Post-processing logic might be incorrect.");
                // คุณอาจจะพ่น Exception ที่นี่ถ้าต้องการการตรวจสอบที่เข้มงวด
            }

        } catch (Exception e) {
            Log.e(TAG, "Error loading LiteRT model from assets: " + modelAssetName + " : " + e.getMessage(), e);
            throw new IOException("Failed to load LiteRT model from assets. Check if org.tensorflow.lite.Interpreter is available.", e);
        }

        if (labelsAssetName != null && !labelsAssetName.isEmpty()) {
            try {
                this.labels = loadLabelsFromAssets(context, labelsAssetName);
                Log.d(TAG, "Labels loaded successfully from assets: " + labelsAssetName + ". Total labels: " + this.labels.size());
                if (this.labels.size() != NUM_CLASSES) {
                    Log.w(TAG, "Mismatch in expected NUM_CLASSES (" + NUM_CLASSES + ") and loaded labels size (" + this.labels.size() + "). Please check labels.txt and NUM_CLASSES constant.");
                }
            } catch (IOException e) {
                Log.e(TAG, "Error loading labels from assets: " + labelsAssetName + " : " + e.getMessage(), e);
            }
        }
    }

    private MappedByteBuffer loadModelFileFromAssets(Context context, String modelAssetName) throws IOException {
        AssetManager assetManager = context.getAssets();
        try (FileInputStream fis = new FileInputStream(assetManager.openFd(modelAssetName).getFileDescriptor())) {
            FileChannel fileChannel = fis.getChannel();
            long startOffset = assetManager.openFd(modelAssetName).getStartOffset();
            long declaredLength = assetManager.openFd(modelAssetName).getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    private List<String> loadLabelsFromAssets(Context context, String labelsAssetName) throws IOException {
        List<String> labels = new ArrayList<>();
        AssetManager assetManager = context.getAssets();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelsAssetName)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        }
        return labels;
    }

    public List<DetectionResult> processImageAndGetResult(Bitmap image) {
        if (interpreter == null) {
            throw new IllegalStateException("LiteRT interpreter not initialized. Check if it loaded correctly.");
        }
        if (image == null) {
            Log.e(TAG, "Input image is null.");
            return new ArrayList<>();
        }

        Bitmap resizedImage = Bitmap.createScaledBitmap(image, modelInputWidth, modelInputHeight, true);

        int numBytesPerChannel = isInputFloat ? 4 : 1;
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(
                modelInputWidth * modelInputHeight * modelInputChannels * numBytesPerChannel
        );
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[modelInputWidth * modelInputHeight];
        resizedImage.getPixels(pixels, 0, modelInputWidth, 0, 0, modelInputWidth, modelInputHeight);

        for (int pixel : pixels) {
            int red = (pixel >> 16) & 0xFF;
            int green = (pixel >> 8) & 0xFF;
            int blue = pixel & 0xFF;

            if (isInputFloat) {
                inputBuffer.putFloat(red / 255.0f);
                inputBuffer.putFloat(green / 255.0f);
                inputBuffer.putFloat(blue / 255.0f);
            } else {
                inputBuffer.put((byte) red);
                inputBuffer.put((byte) green);
                inputBuffer.put((byte) blue);
            }
        }
        inputBuffer.rewind();

        // ขนาดของ Output Buffer ควรเป็น 1 (batch) * 15 (rows) * 4725 (prediction slots) * 4 (bytes per float)
        int outputArraySize = 15 * NUM_PREDICTION_SLOTS; // 15 * 4725 = 70875
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputArraySize * 4); // 4 bytes per float
        outputBuffer.order(ByteOrder.nativeOrder());

        interpreter.run(inputBuffer, outputBuffer);

        outputBuffer.rewind();
        FloatBuffer floatOutputBuffer = outputBuffer.asFloatBuffer();
        float[] rawOutput = new float[outputArraySize];
        floatOutputBuffer.get(rawOutput);

        List<DetectionResult> detectedObjects = new ArrayList<>();

        // ตีความผลลัพธ์จาก rawOutput ตามโครงสร้าง [1, 15, 4725]
        // เราจะวนลูปตามจำนวน Prediction Slots (4725)
        for (int i = 0; i < NUM_PREDICTION_SLOTS; i++) {
            // ดึงพิกัด Bounding Box ในรูปแบบ x_center, y_center, width, height
            float x_center = rawOutput[X_CENTER_COORD_ROW_INDEX * NUM_PREDICTION_SLOTS + i];
            float y_center = rawOutput[Y_CENTER_COORD_ROW_INDEX * NUM_PREDICTION_SLOTS + i];
            float width = rawOutput[WIDTH_COORD_ROW_INDEX * NUM_PREDICTION_SLOTS + i];
            float height = rawOutput[HEIGHT_COORD_ROW_INDEX * NUM_PREDICTION_SLOTS + i];

            // !!! การเปลี่ยนแปลง: แปลงจาก x_center, y_center, width, height เป็น x_min, y_min, x_max, y_max !!!
            float x_min = x_center - (width / 2.0f);
            float y_min = y_center - (height / 2.0f);
            float x_max = x_center + (width / 2.0f);
            float y_max = y_center + (height / 2.0f);

            // !!! การเปลี่ยนแปลง: Clamp ค่าพิกัดให้อยู่ในช่วง [0, 1] !!!
            x_min = Math.max(0.0f, x_min);
            y_min = Math.max(0.0f, y_min);
            x_max = Math.min(1.0f, x_max);
            y_max = Math.min(1.0f, y_max);


            // หา Class Probabilities
            float maxClassProb = -1.0f;
            int predictedClassId = -1;

            // วนลูปผ่าน Class Probabilities (เริ่มจากแถวที่ FIRST_CLASS_PROB_ROW_INDEX ถึง FIRST_CLASS_PROB_ROW_INDEX + NUM_CLASSES - 1)
            for (int k = 0; k < NUM_CLASSES; k++) {
                // Index ของ class probability สำหรับ class k ที่ prediction slot i
                float classProb = rawOutput[(FIRST_CLASS_PROB_ROW_INDEX + k) * NUM_PREDICTION_SLOTS + i];

                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    predictedClassId = k;
                }
            }

            // ใช้ maxClassProb เป็นค่า Confidence Score
            float confidence = maxClassProb;

            // ตรวจสอบ Confidence Threshold
            if (confidence < CONFIDENCE_THRESHOLD) {
                continue; // ข้าม detection ที่มีความมั่นใจต่ำ
            }

            // ตรวจสอบความถูกต้องของ Bounding Box (เช่น x_min < x_max และ y_min < y_max)
            // แม้จะมีการแปลงและ Clamp แล้ว ก็ยังคงการตรวจสอบนี้ไว้เป็น safeguard
            if (x_min >= x_max || y_min >= y_max) {
                Log.w(TAG, "Invalid bounding box dimensions after conversion for prediction slot " + i + ": [" + x_min + ", " + y_min + ", " + x_max + ", " + y_max + "]. Skipping.");
                continue; // ข้าม Bounding Box ที่ไม่ถูกต้อง
            }

            if (predictedClassId != -1 && maxClassProb > 0.0f) {
                String label = (labels != null && predictedClassId < labels.size()) ? labels.get(predictedClassId) : "Unknown_Class_" + predictedClassId;
                detectedObjects.add(new DetectionResult(x_min, y_min, x_max, y_max,
                        confidence, predictedClassId, maxClassProb, label));
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        // Sort detections by confidence score in descending order
        Collections.sort(detectedObjects, new Comparator<DetectionResult>() {
            @Override
            public int compare(DetectionResult d1, DetectionResult d2) {
                return Float.compare(d2.confidence, d1.confidence); // Descending order
            }
        });

        List<DetectionResult> nmsDetections = new ArrayList<>();
        boolean[] isSuppressed = new boolean[detectedObjects.size()];

        for (int i = 0; i < detectedObjects.size(); i++) {
            if (isSuppressed[i]) continue;

            DetectionResult currentDetection = detectedObjects.get(i);
            nmsDetections.add(currentDetection);

            for (int j = i + 1; j < detectedObjects.size(); j++) {
                if (isSuppressed[j]) continue;

                DetectionResult otherDetection = detectedObjects.get(j);

                // Apply NMS only if they predict the same class
                if (currentDetection.classId == otherDetection.classId) {
                    float iou = currentDetection.calculateIoU(otherDetection);
                    if (iou > IOU_THRESHOLD) {
                        isSuppressed[j] = true;
                    }
                }
            }
        }

        if (nmsDetections.isEmpty()) {
            Log.d(TAG, "No objects detected after NMS.");
        } else {
            for (DetectionResult dr : nmsDetections) {
                Log.d(TAG, "Detected: " + dr.toString());
            }
        }

        return nmsDetections;
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            Log.d(TAG, "LiteRT interpreter closed.");
        }
    }
}