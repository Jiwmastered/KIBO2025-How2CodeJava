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

        for (int i = 0; i < sequencePath.size(); i++) {
            String areaName = sequencePath.get(i);
            List<Pair<Point, Quaternion>> areaPosList = pointsMapList.get(areaName);

            if (areaPosList == null || areaPosList.isEmpty()) {
                Log.e(TAG, "No coordinates defined for area: " + areaName + ". Skipping this area.");
                continue;
            }

            Log.i(TAG, "Processing " + areaName);

            List<RobotVisionProcessor.DetectionResult> allDetectionsInCurrentArea = new ArrayList<>();

            for (int j = 0; j < areaPosList.size(); j++) {
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

                try {
                    Bitmap bitmapDockCam = api.getBitmapDockCam();
                    Mat matDockCam = api.getMatDockCam();
                    Bitmap bitmapNavCam = api.getBitmapNavCam();
                    Mat matNavCam = api.getMatNavCam();

                    List<String> detectedLabelsForCurrentImage = new ArrayList<>();

                    if (bitmapNavCam != null) {
                        Bitmap preprocessedNavCamBitmap = bitmapCalibrate(bitmapNavCam);

                        List<RobotVisionProcessor.DetectionResult> detections = visionProcessor.processImageAndGetResult(preprocessedNavCamBitmap);
                        allDetectionsInCurrentArea.addAll(detections);

                        Set<String> uniqueLabels = new HashSet<>();
                        for (RobotVisionProcessor.DetectionResult d : detections) {
                            if (d.confidence >= RobotVisionProcessor.CONFIDENCE_THRESHOLD) {
                                uniqueLabels.add(d.label);
                            }
                        }
                        detectedLabelsForCurrentImage.addAll(uniqueLabels);
                        Collections.sort(detectedLabelsForCurrentImage);

                        for (RobotVisionProcessor.DetectionResult d : detections) {
                            Log.d(TAG, "Detected in " + areaName + " pos " + (j+1) + ": " + d.toString());
                        }

                    } else {
                        Log.w(TAG, "NavCam image was null at " + areaName + " pos " + (j+1) + ". No prediction made.");
                        // แทนที่ api.sendDisablerose ด้วย Log.e
                        Log.e(TAG, "NavCam image null at " + areaName + " pos " + (j+1));
                    }

                    saveImagePack(bitmapDockCam, matDockCam, bitmapNavCam, matNavCam, i, j, detectedLabelsForCurrentImage);

                } catch (Exception e) {
                    Log.e(TAG, "Exception during image capture or vision processing for " + areaName + " position " + (j+1) + ": " + e.getMessage(), e);
                    // แทนที่ api.sendDisablerose ด้วย Log.e
                    Log.e(TAG, "Image processing error at " + areaName + " pos " + (j+1) + ": " + e.getMessage());
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
                    // แทนที่ api.sendDisablerose ด้วย Log.e
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
                    // แทนที่ api.sendDisablerose ด้วย Log.e
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
            // แทนที่ api.sendDisablerose ด้วย Log.e
            Log.e(TAG, "Move to astronaut failed: " + e.getMessage());
        }

        try {
            Bitmap bitmapNavCamAstro = api.getBitmapNavCam();
            Mat matNavCamAstro = api.getMatNavCam();
            List<String> emptyDetectedLabels = new ArrayList<>();
            saveImagePack(null, null, bitmapNavCamAstro, matNavCamAstro, -1, 999, emptyDetectedLabels);
        } catch (Exception e) {
            Log.e(TAG, "Exception during saving astronaut images: " + e.getMessage(), e);
            // แทนที่ api.sendDisablerose ด้วย Log.e
            Log.e(TAG, "Saving astronaut image failed: " + e.getMessage());
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
                    // แทนที่ api.sendDisablerose ด้วย Log.e
                    Log.e(TAG, "Move to target item failed: " + e.getMessage());
                }

                try {
                    api.takeTargetItemSnapshot();
                    Log.i(TAG, "Target item snapshot taken. Mission completed successfully.");
                } catch (Exception e) {
                    Log.e(TAG, "Exception during takeTargetItemSnapshot: " + e.getMessage(), e);
                    // แทนที่ api.sendDisablerose ด้วย Log.e
                    Log.e(TAG, "Taking target item snapshot failed: " + e.getMessage());
                }

            } else {
                Log.e(TAG, "Target coordinates for " + areaTarget + " not found or fewer than 2 points. Cannot move to 2nd target item position.");
                // แทนที่ api.sendDisablerose ด้วย Log.e
                Log.e(TAG, "Target coordinates for " + areaTarget + " not found or insufficient.");
            }
        } else {
            Log.e(TAG, "Target item area not found based on recognition results. Mission may not be fully completed.");
            // แทนที่ api.sendDisablerose ด้วย Log.e
            Log.e(TAG, "Target item area not found in detected classes.");
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

        int imageWidth = distortedImg.getWidth();
        int imageHeight = distortedImg.getHeight();

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
}