package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.util.Log;
import android.util.Pair;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.calib3d.Calib3d;

// เพิ่ม Imports สำหรับ OpenCV ที่จำเป็นในการ cropImage
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect; // ใช้ Rect ของ OpenCV
// สิ้นสุดส่วนเพิ่ม Imports

import android.text.TextUtils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.Objects;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.Comparator; // เพิ่มสำหรับ Collections.sort


/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 * This code is configured to run on Astrobee and perform image recognition using its cameras.
 */
public class YourService extends KiboRpcService {

    private Map<String, List<Pair<Point, Quaternion>>> pointsMapList = new HashMap<>();
    private Map<String, List<String>> areaDetectedClassesMap = new HashMap<>();
    private Pair<Pair<String, String>, String> itemTarget = Pair.create(Pair.create("", ""), "" );

    private Map<String, Integer> areaNameToIdMap;

    private static final String TAG = "AstrobeeMission";
    private RobotVisionProcessor visionProcessor;

    // Check your model and label file names
    private static final String MODEL_ASSET_NAME = "best_float32.tflite";
    private static final String LABELS_ASSET_NAME = "label_txt.txt";
    private static final int MODEL_INPUT_WIDTH = 480;
    private static final int MODEL_INPUT_HEIGHT = 480;
    private static final int MODEL_INPUT_CHANNELS = 3;
    private static final boolean IS_MODEL_INPUT_FLOAT = true;

    private void saveImagePack(Bitmap bitmapDockCam, Mat matDockCam, Bitmap bitmapNavCam, Mat matNavCam, int areaIndex, int imgIndex, List<String> detectedLabels) {
        String labelSuffix = "";
        if (detectedLabels != null && !detectedLabels.isEmpty()) {
            labelSuffix = "_" + TextUtils.join("_", detectedLabels).replace(" ", "_");
        } else {
            labelSuffix = "_no_object_detected";
        }

        if (bitmapDockCam != null) {
            api.saveBitmapImage(bitmapDockCam, "bit_dock_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        } else {
            Log.w(TAG, "DockCam bitmap is null, skipping save for: bit_dock_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        }
        if (matDockCam != null) {
            api.saveMatImage(matDockCam, "mat_dock_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        } else {
            Log.w(TAG, "DockCam Mat is null, skipping save for: mat_dock_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        }
        // Note: bitmapNavCam and matNavCam will be saved here if not null,
        // but the calibrated/cropped versions are saved separately below
        if (bitmapNavCam != null) {
            api.saveBitmapImage(bitmapNavCam, "bit_nav_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        } else {
            Log.w(TAG, "NavCam bitmap is null, skipping save for: bit_nav_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        }
        if (matNavCam != null) {
            api.saveMatImage(matNavCam, "mat_nav_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        } else {
            Log.w(TAG, "NavCam Mat is null, skipping save for: mat_nav_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "YourService onCreate started. Initializing RobotVisionProcessor.");

        areaNameToIdMap = new HashMap<>();
        areaNameToIdMap.put("Area 1", 1);
        areaNameToIdMap.put("Area 2", 2);
        areaNameToIdMap.put("Area 3", 3);
        areaNameToIdMap.put("Area 4", 4);

        try {
            visionProcessor = new RobotVisionProcessor(
                    this,
                    MODEL_ASSET_NAME,
                    LABELS_ASSET_NAME,
                    MODEL_INPUT_WIDTH,
                    MODEL_INPUT_HEIGHT,
                    MODEL_INPUT_CHANNELS,
                    IS_MODEL_INPUT_FLOAT
            );
            Log.d(TAG, "RobotVisionProcessor initialized successfully.");
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize RobotVisionProcessor: " + e.getMessage(), e);
            // แทนที่ api.sendDisablerose ด้วย Log.e
            Log.e(TAG, "Vision processor initialization failed: " + e.getMessage());
        }
    }

    private void sendResultToRobot(String result) {
        if (result == null || result.isEmpty()) {
            Log.w(TAG, "Attempted to send null or empty result. Ignoring.");
            return;
        }
        Log.d(TAG, "Attempting to send result back to Astrobee: " + result);
        Log.i(TAG, "Placeholder: Result for Astrobee communication: " + result);
    }

    private List<String> getUniqueDetectedItems(List<RobotVisionProcessor.DetectionResult> allDetectionsInArea) {
        if (allDetectionsInArea == null || allDetectionsInArea.isEmpty()) {
            Log.w(TAG, "No detections provided for aggregation.");
            return new ArrayList<>();
        }

        Set<String> uniqueItems = new HashSet<>();
        for (RobotVisionProcessor.DetectionResult detection : allDetectionsInArea) {
            if (detection.confidence >= RobotVisionProcessor.CONFIDENCE_THRESHOLD) {
                uniqueItems.add(detection.label);
            }
        }
        List<String> result = new ArrayList<>(uniqueItems);
        Collections.sort(result);

        Log.d(TAG, "Aggregated unique items for area: " + result.toString());
        return result;
    }


    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "YourService onDestroy started.");

        if (visionProcessor != null) {
            visionProcessor.close();
            Log.d(TAG, "RobotVisionProcessor closed.");
        }
    }


    @Override
    protected void runPlan1() {
        api.startMission();
        Log.i(TAG, "Mission started. Astrobee is ready!");

        if (visionProcessor == null) {
            Log.e(TAG, "RobotVisionProcessor was not initialized successfully. Cannot proceed with vision tasks.");
            // แทนที่ api.sendDisablerose ด้วย Log.e
            Log.e(TAG, "Vision system failed to initialize. Mission aborted.");
            return;
        }

        pointsMapList.put("Area 1", new ArrayList<Pair<Point, Quaternion>>(Arrays.asList(
                new Pair<>(new Point(10.6265d, -10.0406d, 4.75906d), new Quaternion(-0.176166f, -0.176166f, 0.684811f, 0.684811f)),
                new Pair<>(new Point(10.9211d, -10.0406d, 4.75906d), new Quaternion(-0.176166f, -0.176166f, 0.684811f, 0.684811f)),
                new Pair<>(new Point(11.2711d, -10.0406d, 4.75906d), new Quaternion(-0.176166f, -0.176166f, 0.684811f, 0.684811f))
        )));

        pointsMapList.put("Area 2", new ArrayList<Pair<Point, Quaternion>>(Arrays.asList(
                new Pair<>(new Point(11.3432d, -8.92783d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(10.9358d, -8.92783d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(10.5544d, -8.92783d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f))
        )));

        pointsMapList.put("Area 3", new ArrayList<Pair<Point, Quaternion>>(Arrays.asList(
                new Pair<>(new Point(10.5602d, -7.96923d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(10.9503d, -7.96923d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(11.3403d, -7.96923d, 4.45397d), new Quaternion(0f, 0f, 0.707107f, 0.707107f))
        )));

        pointsMapList.put("Area 4", new ArrayList<Pair<Point, Quaternion>>(Arrays.asList(
                new Pair<>(new Point(10.6149d, -6.82423d, 4.55599d), new Quaternion(0f, 0f, 1f, 0f)),
                new Pair<>(new Point(10.6149d, -6.82423d, 4.91140d), new Quaternion(0f, 0f, 1f, 0f)),
                new Pair<>(new Point(10.6149d, -6.82423d, 5.31016d), new Quaternion(0f, 0f, 1f, 0f))
        )));

        List<String> sequencePath = new ArrayList<>(Arrays.asList("Area 1", "Area 2", "Area 3", "Area 4"));

        for (int i = 0; i < sequencePath.size(); i++) { // i is areaIndex
            String areaName = sequencePath.get(i);
            List<Pair<Point, Quaternion>> areaPosList = pointsMapList.get(areaName);

            if (areaPosList == null || areaPosList.isEmpty()) {
                Log.e(TAG, "No coordinates defined for area: " + areaName + ". Skipping this area.");
                continue;
            }

            Log.i(TAG, "Processing " + areaName);

            List<RobotVisionProcessor.DetectionResult> allDetectionsInCurrentArea = new ArrayList<>();

            for (int j = 0; j < areaPosList.size(); j++) { // j is imgIndex
                Pair<Point, Quaternion> move = areaPosList.get(j);
                Point point = move.first;
                Quaternion quaternion = move.second;

                Log.d(TAG, "Attempting to move to position " + (j + 1) + " in " + areaName + ": X=" + point.getX() + ", Y=" + point.getY() + ", Z=" + point.getZ());
                try {
                    api.moveTo(point, quaternion, false);
                    Log.d(TAG, "Move command sent to " + areaName + " position " + (j+1) + ". Check robot logs for actual success.");
                } catch (Exception e) {
                    Log.e(TAG, "Exception during moveTo command for " + areaName + " position " + (j+1) + ": " + e.getMessage(), e);
                    // แทนที่ api.sendDisablerose ด้วย Log.e
                    Log.e(TAG, "Move failed for " + areaName + " pos " + (j+1) + ": " + e.getMessage());
                    continue;
                }

                Bitmap bitmapDockCam = null;
                Mat matDockCam = null;
                Bitmap bitmapNavCam = null;
                Mat matNavCam = null;

                try {
                    bitmapDockCam = api.getBitmapDockCam();
                    matDockCam = api.getMatDockCam();
                    bitmapNavCam = api.getBitmapNavCam();
                    matNavCam = api.getMatNavCam();

                    List<String> detectedLabelsForCurrentImage = new ArrayList<>(); // This will store labels for naming current image saves

                    if (bitmapNavCam != null) {
                        // 1. Get calibrated image
                        Bitmap preprocessedNavCamBitmap = bitmapCalibrate(bitmapNavCam);

                        // 2. Process image and get detections
                        List<RobotVisionProcessor.DetectionResult> detections = visionProcessor.processImageAndGetResult(preprocessedNavCamBitmap);
                        allDetectionsInCurrentArea.addAll(detections); // Accumulate all detections for the area

                        // Prepare labels for naming (for calibrated and cropped images)
                        Set<String> uniqueLabelsForImageSet = new HashSet<>(); // Use a set to collect unique labels for the current image
                        for (RobotVisionProcessor.DetectionResult d : detections) {
                            if (d.confidence >= RobotVisionProcessor.CONFIDENCE_THRESHOLD) {
                                uniqueLabelsForImageSet.add(d.label); // Add to set if confident
                            }
                        }
                        List<String> sortedUniqueLabelsForImage = new ArrayList<>(uniqueLabelsForImageSet);
                        Collections.sort(sortedUniqueLabelsForImage);

                        String currentImageLabelSuffix = "";
                        if (!sortedUniqueLabelsForImage.isEmpty()) {
                            currentImageLabelSuffix = "_" + TextUtils.join("_", sortedUniqueLabelsForImage).replace(" ", "_");
                        } else {
                            currentImageLabelSuffix = "_no_object_detected";
                        }

                        // 3. Save the CALIBRATED image
                        // Note: Using a fixed filename for the calibrated image per area/img,
                        // this assumes you only want one calibrated image saved per position.
                        api.saveBitmapImage(preprocessedNavCamBitmap, "calibrate_" + i + "_" + j + currentImageLabelSuffix);
                        Log.d(TAG, "Saved calibrated image: calibrate_" + i + "_" + j + currentImageLabelSuffix);


                        // 4. Perform the contour-based crop and save
                        Bitmap contourCroppedBitmap = cropImage(preprocessedNavCamBitmap);
                        if (contourCroppedBitmap != null) {
                            // ใช้ "calibrate_crop_" + areaIndex + "_" + imgIndex + labelSuffix
                            String contourCroppedFileName = "calibrate_crop_" + i + "_" + j + currentImageLabelSuffix + "_contour_new"; // เพิ่ม "_contour_new" เพื่อความชัดเจนและแยกแยะ
                            api.saveBitmapImage(contourCroppedBitmap, contourCroppedFileName);
                            Log.d(TAG, "Saved contour-cropped image: " + contourCroppedFileName);
                            contourCroppedBitmap.recycle(); // IMPORTANT: Recycle the cropped bitmap
                        } else {
                            Log.w(TAG, "Failed to perform contour-based crop for image at " + areaName + " pos " + (j+1));
                        }


                        // Update detectedLabelsForCurrentImage for the saveImagePack method (for raw images)
                        detectedLabelsForCurrentImage.addAll(sortedUniqueLabelsForImage);

                        // 5. Log detections (already existing)
                        for (RobotVisionProcessor.DetectionResult d : detections) {
                            Log.d(TAG, "Detected in " + areaName + " pos " + (j+1) + ": " + d.toString());
                        }

                        // 6. Recycle the preprocessedNavCamBitmap after all processing is done for it
                        if (preprocessedNavCamBitmap != null) {
                            preprocessedNavCamBitmap.recycle();
                        }

                    } else {
                        Log.w(TAG, "NavCam image was null at " + areaName + " pos " + (j+1) + ". No prediction made.");
                        Log.e(TAG, "NavCam image null at " + areaName + " pos " + (j+1));
                    }

                    // Save the original (raw) images using the existing saveImagePack method
                    saveImagePack(bitmapDockCam, matDockCam, bitmapNavCam, matNavCam, i, j, detectedLabelsForCurrentImage);

                } catch (Exception e) {
                    Log.e(TAG, "Exception during image capture or vision processing for " + areaName + " position " + (j+1) + ": " + e.getMessage(), e);
                    Log.e(TAG, "Image processing error at " + areaName + " pos " + (j+1) + ": " + e.getMessage());
                } finally {
                    // Ensure all Bitmaps and Mats obtained from API are recycled/released
                    // regardless of exceptions to prevent memory leaks
                    if (bitmapNavCam != null) {
                        bitmapNavCam.recycle();
                    }
                    if (bitmapDockCam != null) {
                        bitmapDockCam.recycle();
                    }
                    if (matNavCam != null) {
                        matNavCam.release();
                    }
                    if (matDockCam != null) {
                        matDockCam.release();
                    }
                }
            }

            List<String> uniqueDetectedItems = getUniqueDetectedItems(allDetectionsInCurrentArea);

            if (!uniqueDetectedItems.isEmpty()) {
                Log.i(TAG, "Final unique detected items for " + areaName + ": " + uniqueDetectedItems.toString());

                areaDetectedClassesMap.put(areaName, uniqueDetectedItems);

                String reportString = TextUtils.join(",", uniqueDetectedItems);
                if (reportString.isEmpty() && !allDetectionsInCurrentArea.isEmpty()) {
                    reportString = "Detected_But_Low_Conf";
                } else if (reportString.isEmpty()) {
                    reportString = "No_Item_Detected";
                }

                Integer areaId = areaNameToIdMap.get(areaName);
                if (areaId != null) {
                    api.setAreaInfo(areaId, reportString);
                    Log.i(TAG, "Reported Area: " + areaName + " (ID: " + areaId + "), Detected Items: " + reportString);
                } else {
                    Log.e(TAG, "Area ID not found for areaName: " + areaName + ". Cannot report area info via API.");
                    Log.e(TAG, "Area ID not found for " + areaName + ". Reporting skipped.");
                }

            } else {
                Log.w(TAG, "No valid objects detected for " + areaName + ". Reporting 'No_Item_Detected'.");
                Integer areaId = areaNameToIdMap.get(areaName);
                if (areaId != null) {
                    api.setAreaInfo(areaId, "No_Item_Detected");
                    Log.i(TAG, "Reported Area: " + areaName + " (ID: " + areaId + "), Detected Items: No_Item_Detected");
                } else {
                    Log.e(TAG, "Area ID not found for areaName: " + areaName + ". Cannot report default item.");
                    Log.e(TAG, "Area ID not found for " + areaName + ". Reporting skipped.");
                }
            }
        }

        Log.i(TAG, "All patrol areas visited. Moving to astronaut.");

        Point pointAstro = new Point(11.143d, -6.7607d, 4.9654d);
        Quaternion quaternionAstro = new Quaternion(0f, 0f, 0.707f, 0.707f);
        try {
            api.moveTo(pointAstro, quaternionAstro, false);
            Log.i(TAG, "Move command sent to Astronaut position. Check robot logs for actual success.");
        } catch (Exception e) {
            Log.e(TAG, "Exception during move to Astronaut position: " + e.getMessage(), e);
            Log.e(TAG, "Move to astronaut failed: " + e.getMessage());
        }

        Bitmap bitmapNavCamAstro = null;
        Mat matNavCamAstro = null;
        try {
            bitmapNavCamAstro = api.getBitmapNavCam();
            matNavCamAstro = api.getMatNavCam();

            // Calibrate the astronaut image
            Bitmap preprocessedNavCamBitmapAstro = null;
            if (bitmapNavCamAstro != null) {
                preprocessedNavCamBitmapAstro = bitmapCalibrate(bitmapNavCamAstro);
                api.saveBitmapImage(preprocessedNavCamBitmapAstro, "calibrate_astro_999");
            }

            // Crop the astronaut image using the new crop_image logic
            Bitmap croppedNavCamBitmapAstro = null;
            if (preprocessedNavCamBitmapAstro != null) {
                croppedNavCamBitmapAstro = cropImage(preprocessedNavCamBitmapAstro);
                api.saveBitmapImage(croppedNavCamBitmapAstro, "calibrate_crop_astro_999_new"); // Save with a specific name
            } else {
                Log.w(TAG, "Preprocessed astronaut bitmap is null, cannot perform crop.");
            }

            List<String> emptyDetectedLabels = new ArrayList<>();
            // Save the original raw astronaut images
            saveImagePack(null, null, bitmapNavCamAstro, matNavCamAstro, -1, 999, emptyDetectedLabels);

            // Recycle bitmaps
            if (preprocessedNavCamBitmapAstro != null) {
                preprocessedNavCamBitmapAstro.recycle();
            }
            if (croppedNavCamBitmapAstro != null) {
                croppedNavCamBitmapAstro.recycle();
            }

        } catch (Exception e) {
            Log.e(TAG, "Exception during saving astronaut images: " + e.getMessage(), e);
            Log.e(TAG, "Saving astronaut image failed: " + e.getMessage());
        } finally {
            if (bitmapNavCamAstro != null) {
                bitmapNavCamAstro.recycle();
            }
            if (matNavCamAstro != null) {
                matNavCamAstro.release();
            }
        }

        api.reportRoundingCompletion();
        Log.i(TAG, "Rounding completion reported.");

        itemTarget = Pair.create(Pair.create("your_target_landmark_1", "your_target_landmark_2"), "your_target_treasure");


        String areaTarget = "";
        Log.i(TAG, "Searching for target item: " + itemTarget.first.first + ", " + itemTarget.first.second + " with treasure: " + itemTarget.second);
        for (Map.Entry<String, List<String>> entry : areaDetectedClassesMap.entrySet()) {
            String currentAreaName = entry.getKey();
            List<String> detectedClassesInArea = entry.getValue();

            Log.d(TAG, "Checking Area: " + currentAreaName + ", Detected Classes: " + detectedClassesInArea);

            boolean landmarkMatch = false;
            if (!Objects.equals(itemTarget.first.first, "") && detectedClassesInArea.contains(itemTarget.first.first)) {
                landmarkMatch = true;
            }
            if (!Objects.equals(itemTarget.first.second, "") && detectedClassesInArea.contains(itemTarget.first.second)) {
                landmarkMatch = true;
            }

            boolean treasureMatch = Objects.equals(itemTarget.second, "") || detectedClassesInArea.contains(itemTarget.second);

            if (landmarkMatch && treasureMatch) {
                areaTarget = currentAreaName;
                Log.i(TAG, "Target item found in " + areaTarget + "!");
                break;
            }
        }

        api.notifyRecognitionItem();
        if (!areaTarget.isEmpty()) {

            Log.i(TAG, "Recognition item notified for target area: " + areaTarget);

            List<Pair<Point, Quaternion>> targetCoordList = pointsMapList.get(areaTarget);
            if (targetCoordList != null && targetCoordList.size() >= 2) {
                Pair<Point, Quaternion> targetCoordPair = targetCoordList.get(1);
                Point point = targetCoordPair.first;
                Quaternion quaternion = targetCoordPair.second;

                Log.i(TAG, "Moving to the 2nd coordinate of target area " + areaTarget + " at: X=" + point.getX() + ", Y=" + point.getY() + ", Z=" + point.getZ());
                try {
                    api.moveTo(point, quaternion, false);
                    Log.i(TAG, "Move command sent to target item position. Check robot logs for actual success.");
                } catch (Exception e) {
                    Log.e(TAG, "Exception during move to target item position (" + areaTarget + "): " + e.getMessage(), e);
                }

                try {
                    api.takeTargetItemSnapshot();
                    Log.i(TAG, "Target item snapshot taken. Mission completed successfully.");
                } catch (Exception e) {
                    Log.e(TAG, "Exception during takeTargetItemSnapshot: " + e.getMessage(), e);
                }

            } else {
                Log.e(TAG, "Target coordinates for " + areaTarget + " not found or fewer than 2 points. Cannot move to 2nd target item position.");
            }
        } else {
            Log.e(TAG, "Target item area not found based on recognition results. Mission may not be fully completed.");
        }
    }

    private static void assignMat(Mat mat, Double[][] arr) {
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                mat.put(i, j, arr[i][j]);
            }
        }
    }

    public static Bitmap bitmapCalibrate(Bitmap distortedImg) {
        Double NavCamArr[][] = {
                {523.105750, 0.000000, 635.434258},
                {0.000000, 534.765913, 500.335102},
                {0.000000, 0.000000, 1.000000}
        };
        Double DistCamArr[][] = {{-0.164787, 0.020375, -0.001572, -0.000369, 0.000000}};

        Mat simNavCamMatrix = new Mat(3, 3, CvType.CV_64FC1);
        assignMat(simNavCamMatrix, NavCamArr);
        Mat simNavCamDistort = new Mat(1, 5, CvType.CV_64FC1);
        assignMat(simNavCamDistort, DistCamArr);

        Mat distortedImageMat = new Mat();
        Utils.bitmapToMat(distortedImg, distortedImageMat);
        Mat undistortedImageMat = new Mat();

        Calib3d.undistort(distortedImageMat, undistortedImageMat, simNavCamMatrix, simNavCamDistort);

        Bitmap undistortedImageBitmap = Bitmap.createBitmap(undistortedImageMat.cols(), undistortedImageMat.rows(), distortedImg.getConfig());
        Utils.matToBitmap(undistortedImageMat, undistortedImageBitmap);

        simNavCamMatrix.release();
        simNavCamDistort.release();
        distortedImageMat.release();
        undistortedImageMat.release();

        return undistortedImageBitmap;
    }

    /**
     * Crops an image based on the largest 4-point contour found.
     * Implemented based on the provided Python code.
     * @param inputBitmap The input Bitmap to be cropped.
     * @return The cropped Bitmap, or the original Bitmap if no valid contour is found or cropping fails.
     */
    public static Bitmap cropImage(Bitmap inputBitmap) {
        Mat img = new Mat();
        Utils.bitmapToMat(inputBitmap, img);
        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        Mat thresh = new Mat();
        Imgproc.adaptiveThreshold(gray, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);;

        Mat hsv = new Mat();
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV);
        Scalar lowerWhite = new Scalar(0, 0, 160);
        Scalar upperWhite = new Scalar(180, 20, 255);
        Mat mask = new Mat();
        Core.inRange(hsv, lowerWhite, upperWhite, mask);

        Mat filtered = new Mat();
        Imgproc.GaussianBlur(thresh, filtered, new org.opencv.core.Size(5, 5), 0);

        Mat edged = new Mat();
        Imgproc.Canny(filtered, edged, 60, 180);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(3, 3));
        Imgproc.morphologyEx(edged, edged, Imgproc.MORPH_CLOSE, kernel);
        kernel.release(); // Release kernel

        List<MatOfPoint> keypoints = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edged.clone(), keypoints, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release(); // Release hierarchy

        // Sort contours by area in descending order and take top 10
        List<MatOfPoint> contours = new ArrayList<>();
        if (!keypoints.isEmpty()) {
            // Sort by area and add to a new list
            Collections.sort(keypoints, new Comparator<MatOfPoint>() {
                @Override
                public int compare(MatOfPoint o1, MatOfPoint o2) {
                    return Double.compare(Imgproc.contourArea(o2), Imgproc.contourArea(o1));
                }
            });
            for (int k = 0; k < Math.min(keypoints.size(), 10); k++) {
                contours.add(keypoints.get(k));
            }
        }
        // Release original keypoints list Mats
        for (MatOfPoint mp : keypoints) {
            mp.release();
        }

        MatOfPoint location = null;
        for (MatOfPoint contour : contours) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            double epsilon = 0.01 * Imgproc.arcLength(contour2f, true);
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true);

            if (approxCurve.total() == 4) {
                location = new MatOfPoint(approxCurve.toArray());
                approxCurve.release(); // Release approxCurve as its points are copied
                contour2f.release(); // Release contour2f
                break;
            }
            approxCurve.release(); // Release approxCurve for contours not matching 4 points
            contour2f.release(); // Release contour2f for contours not matching 4 points
        }
        // Release remaining contours in the sorted list
        for (MatOfPoint mp : contours) {
            mp.release();
        }


        if (location == null) {
            Log.w(TAG, "cropImage: Could not find a 4-point contour. Returning original image.");
            // Release all intermediate Mats
            img.release();
            gray.release();
            thresh.release();
            hsv.release();
            mask.release();
            filtered.release();
            edged.release();
            return inputBitmap;
        }

        Mat polyMask = Mat.zeros(gray.size(), CvType.CV_8UC1);
        List<MatOfPoint> listLocation = new ArrayList<>();
        listLocation.add(location);
        Imgproc.fillPoly(polyMask, listLocation, new Scalar(255));

        Mat sampleImage = new Mat();
        Core.bitwise_and(img, img, sampleImage, polyMask);

        // Convert MatOfPoint (location) to an array of Point for easier access to x_min, x_max, etc.
        org.opencv.core.Point[] points = location.toArray();

        int x_min = img.cols();
        int x_max = 0;
        int y_min = img.rows();
        int y_max = 0;

        for (org.opencv.core.Point p : points) {
            if (p.x < x_min) x_min = (int) p.x;
            if (p.x > x_max) x_max = (int) p.x;
            if (p.y < y_min) y_min = (int) p.y;
            if (p.y > y_max) y_max = (int) p.y;
        }

        // Add some padding if desired, but ensure it doesn't go out of bounds
        int padding = 5; // Example padding
        x_min = Math.max(0, x_min - padding);
        y_min = Math.max(0, y_min - padding);
        x_max = Math.min(img.cols(), x_max + padding);
        y_max = Math.min(img.rows(), y_max + padding);


        // Check for valid crop dimensions (must be at least 1x1 pixel)
        if (x_max <= x_min || y_max <= y_min) {
            Log.w(TAG, "Invalid ROI for cropping. Returning original image.");
            // Release all intermediate Mats
            img.release();
            gray.release();
            thresh.release();
            hsv.release();
            mask.release();
            filtered.release();
            edged.release();
            polyMask.release();
            sampleImage.release();
            location.release(); // Release location
            return inputBitmap;
        }

        Rect roi = new Rect(x_min, y_min, x_max - x_min, y_max - y_min);
        Mat croppedMat = new Mat(sampleImage, roi);
        Bitmap croppedBitmap = Bitmap.createBitmap(croppedMat.cols(), croppedMat.rows(), inputBitmap.getConfig());
        Utils.matToBitmap(croppedMat, croppedBitmap);

        // Release all Mats created in this method
        img.release();
        gray.release();
        thresh.release();
        hsv.release();
        mask.release();
        filtered.release();
        edged.release();
        polyMask.release();
        sampleImage.release();
        croppedMat.release();
        location.release(); // Release location Mat after use

        return croppedBitmap;
    }
}