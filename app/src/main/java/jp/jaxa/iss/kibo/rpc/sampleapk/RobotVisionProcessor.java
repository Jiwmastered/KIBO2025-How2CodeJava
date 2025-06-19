package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

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

    private static final int NUM_PREDICTION_SLOTS = 4725;

    private static final int X_CENTER_COORD_ROW_INDEX = 0;
    private static final int Y_CENTER_COORD_ROW_INDEX = 1;
    private static final int WIDTH_COORD_ROW_INDEX = 2;
    private static final int HEIGHT_COORD_ROW_INDEX = 3;
    private static final int FIRST_CLASS_PROB_ROW_INDEX = 4;

    public static final float CONFIDENCE_THRESHOLD = 0.5f;
    private static final float IOU_THRESHOLD = 0.45f;
    private static final int NUM_CLASSES = 11;

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
            if (outputShape.length != 3 || outputShape[0] != 1 || outputShape[1] != (4 + NUM_CLASSES) || outputShape[2] != NUM_PREDICTION_SLOTS) {
                Log.e(TAG, "Mismatch in expected model output tensor shape! Expected [1, " + (4 + NUM_CLASSES) + ", " + NUM_PREDICTION_SLOTS + "] but got " + Arrays.toString(outputShape) + ". Post-processing logic might be incorrect.");
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

        // === START: Integration of cropImage method ===
        // 1. Crop the input image using the OpenCV contour-based method.
        // If it fails to find a 4-sided contour, it will return the original image.
        // Note: This cropImage method will operate on the entire 'image'
        // and does not use the specific bounding box from individual detections.
        Bitmap croppedImage = cropImage(image);
        Log.d(TAG, "Image cropping complete. Pre-crop size: " + image.getWidth() + "x" + image.getHeight() +
                ". Post-crop size: " + croppedImage.getWidth() + "x" + croppedImage.getHeight());


        // 2. Resize the (potentially cropped) image to the model's required input dimensions.
        Bitmap resizedImage = Bitmap.createScaledBitmap(croppedImage, modelInputWidth, modelInputHeight, true);

        // 3. Recycle the intermediate cropped bitmap if it was newly created (i.e., not the same as the input image)
        if (croppedImage != image) { // Only recycle if it's a new bitmap, not the input one
            croppedImage.recycle();
        }
        // === END: Integration of cropImage method ===

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

        int outputArraySize = 15 * NUM_PREDICTION_SLOTS;
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputArraySize * 4);
        outputBuffer.order(ByteOrder.nativeOrder());

        interpreter.run(inputBuffer, outputBuffer);

        outputBuffer.rewind();
        FloatBuffer floatOutputBuffer = outputBuffer.asFloatBuffer();
        float[] rawOutput = new float[outputArraySize];
        floatOutputBuffer.get(rawOutput);

        List<DetectionResult> detectedObjects = new ArrayList<>();

        for (int i = 0; i < NUM_PREDICTION_SLOTS; i++) {
            float x_center = rawOutput[X_CENTER_COORD_ROW_INDEX * NUM_PREDICTION_SLOTS + i];
            float y_center = rawOutput[Y_CENTER_COORD_ROW_INDEX * NUM_PREDICTION_SLOTS + i];
            float width = rawOutput[WIDTH_COORD_ROW_INDEX * NUM_PREDICTION_SLOTS + i];
            float height = rawOutput[HEIGHT_COORD_ROW_INDEX * NUM_PREDICTION_SLOTS + i];

            float x_min = x_center - (width / 2.0f);
            float y_min = y_center - (height / 2.0f);
            float x_max = x_center + (width / 2.0f);
            float y_max = y_center + (height / 2.0f);

            x_min = Math.max(0.0f, x_min);
            y_min = Math.max(0.0f, y_min);
            x_max = Math.min(1.0f, x_max);
            y_max = Math.min(1.0f, y_max);

            float maxClassProb = -1.0f;
            int predictedClassId = -1;

            for (int k = 0; k < NUM_CLASSES; k++) {
                float classProb = rawOutput[(FIRST_CLASS_PROB_ROW_INDEX + k) * NUM_PREDICTION_SLOTS + i];

                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    predictedClassId = k;
                }
            }

            float confidence = maxClassProb;

            if (confidence < CONFIDENCE_THRESHOLD) {
                continue;
            }

            if (x_min >= x_max || y_min >= y_max) {
                Log.w(TAG, "Invalid bounding box dimensions after conversion for prediction slot " + i + ": [" + x_min + ", " + y_min + ", " + x_max + ", " + y_max + "]. Skipping.");
                continue;
            }

            if (predictedClassId != -1 && maxClassProb > 0.0f) {
                String label = (labels != null && predictedClassId < labels.size()) ? labels.get(predictedClassId) : "Unknown_Class_" + predictedClassId;
                detectedObjects.add(new DetectionResult(x_min, y_min, x_max, y_max,
                        confidence, predictedClassId, maxClassProb, label));
            }
        }

        Collections.sort(detectedObjects, new Comparator<DetectionResult>() {
            @Override
            public int compare(DetectionResult d1, DetectionResult d2) {
                return Float.compare(d2.confidence, d1.confidence);
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

        // Recycle resizedImage
        if (resizedImage != null) {
            resizedImage.recycle();
        }

        return nmsDetections;
    }

    /**
     * Finds a 4-point contour in the image and crops to its bounding box.
     * This is useful for isolating a document or screen from its background.
     *
     * @param inputBitmap The original Bitmap image.
     * @return A new, cropped Bitmap. If no suitable contour is found, returns the original inputBitmap.
     */
    private static Bitmap cropImage(Bitmap inputBitmap) {
        Mat img = new Mat();
        Utils.bitmapToMat(inputBitmap, img);
        // Convert to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        // Adaptive threshold
        Mat thresh = new Mat();
        Imgproc.adaptiveThreshold(gray, thresh, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY, 11, 2);

        // Convert to HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV);

        // Mask white colors in HSV
        Scalar lowerWhite = new Scalar(0, 0, 160);
        Scalar upperWhite = new Scalar(180, 30, 255);
        Mat mask = new Mat();
        Core.inRange(hsv, lowerWhite, upperWhite, mask);

        // Combine mask with thresholded image
        Mat edged = new Mat();
        Core.bitwise_and(thresh, mask, edged);

        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edged.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Release intermediate Mats
        gray.release();
        thresh.release();
        hsv.release();
        mask.release();
        edged.release();
        hierarchy.release();

        if (contours.isEmpty()) {
            Log.w(TAG, "cropImage: No contours found. Returning original image.");
            img.release();
            return inputBitmap; // Return original inputBitmap as no contour was found
        }

        // Sort contours by area
        Collections.sort(contours, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint a, MatOfPoint b) {
                return Double.compare(Imgproc.contourArea(b), Imgproc.contourArea(a));
            }
        });

        // Find a 4-point contour
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        MatOfPoint location = null;

        for (MatOfPoint contour : contours) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            double epsilon = 0.01 * Imgproc.arcLength(contour2f, true);
            Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true);
            if (approxCurve.total() == 4) {
                location = new MatOfPoint(approxCurve.toArray());
                contour2f.release(); // Release this MatOfPoint2f
                break;
            }
            contour2f.release(); // Release this MatOfPoint2f
        }

        // Release MatOfPoint objects in contours list
        for (MatOfPoint contour : contours) {
            contour.release();
        }

        if (location == null) {
            Log.w(TAG, "cropImage: Could not find a 4-point contour. Returning original image.");
            approxCurve.release();
            img.release();
            return inputBitmap; // Return original inputBitmap as no 4-point contour was found
        }

        // Create mask from polygon
        Mat polyMask = Mat.zeros(img.size(), CvType.CV_8UC1);
        List<MatOfPoint> list = new ArrayList<>();
        list.add(location);
        Imgproc.fillPoly(polyMask, list, new Scalar(255));

        // Apply mask to image
        Mat sampleImage = new Mat();
        Core.bitwise_and(img, img, sampleImage, polyMask);

        // Crop using bounding box of polygon
        // Using org.opencv.core.Rect for OpenCV operations
        org.opencv.core.Rect roi = Imgproc.boundingRect(location);

        // Ensure valid ROI coordinates (within image bounds)
        // This is crucial if boundingRect goes slightly out of bounds or provides invalid dimensions.
        int xMin = Math.max(0, roi.x);
        int yMin = Math.max(0, roi.y);
        int xMax = Math.min(img.cols(), roi.x + roi.width);
        int yMax = Math.min(img.rows(), roi.y + roi.height);

        // Check for valid crop dimensions (must be at least 1x1 pixel)
        if (xMax <= xMin || yMax <= yMin) {
            Log.w(TAG, "Invalid ROI for cropping after bounding box calculation. Returning original image.");
            // Release remaining Mats before returning
            img.release();
            approxCurve.release();
            location.release();
            polyMask.release();
            sampleImage.release();
            return inputBitmap;
        }

        org.opencv.core.Rect safeRoi = new org.opencv.core.Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        Mat croppedMat = new Mat(sampleImage, safeRoi);
        Bitmap croppedBitmap = Bitmap.createBitmap(croppedMat.cols(), croppedMat.rows(), inputBitmap.getConfig());
        Utils.matToBitmap(croppedMat, croppedBitmap);

        // Release all remaining Mats
        img.release();
        approxCurve.release();
        location.release();
        polyMask.release();
        sampleImage.release();
        croppedMat.release();

        return croppedBitmap;
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            Log.d(TAG, "LiteRT interpreter closed.");
        }
    }
}