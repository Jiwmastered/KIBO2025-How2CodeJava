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

import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;

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
import java.util.Comparator;


/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 * This code is configured to run on Astrobee and perform image recognition using its cameras.
 */
public class YourService extends KiboRpcService {

    private Map<String, List<Pair<Point, Quaternion>>> pointsMapList = new HashMap<>();
    private Map<String, List<String>> areaDetectedClassesMap = new HashMap<>();
    private Pair<Pair<String, String>, String> itemTarget = Pair.create(Pair.create("", ""), "");

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

    // แก้ไข saveImagePack ให้เลือก save Dock Cam หรือ Nav Cam ได้ และไม่เซฟ Mat
    private void saveImagePack(Bitmap bitmapDockCam, Bitmap bitmapNavCam, int areaIndex, int imgIndex, List<String> detectedLabels, boolean saveDockCam, boolean saveNavCam) {
        String labelSuffix = "";
        if (detectedLabels != null && !detectedLabels.isEmpty()) {
            labelSuffix = "_" + TextUtils.join("_", detectedLabels).replace(" ", "_");
        } else {
            labelSuffix = "_no_object_detected";
        }

        if (saveDockCam && bitmapDockCam != null) {
            api.saveBitmapImage(bitmapDockCam, "bit_dock_area_" + areaIndex + "_" + imgIndex + labelSuffix);
            Log.d(TAG, "Saved RAW DockCam Bitmap: bit_dock_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        } else if (saveDockCam && bitmapDockCam == null) {
            Log.w(TAG, "DockCam RAW bitmap is null, skipping save for: bit_dock_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        }

        if (saveNavCam && bitmapNavCam != null) {
            api.saveBitmapImage(bitmapNavCam, "bit_nav_area_" + areaIndex + "_" + imgIndex + labelSuffix);
            Log.d(TAG, "Saved RAW NavCam Bitmap: bit_nav_area_" + areaIndex + "_" + imgIndex + labelSuffix);
        } else if (saveNavCam && bitmapNavCam == null) {
            Log.w(TAG, "NavCam RAW bitmap is null, skipping save for: bit_nav_area_" + areaIndex + "_" + imgIndex + labelSuffix);
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
            Log.e(TAG, "Vision processor initialization failed: " + e.getMessage());
        }
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
            Log.e(TAG, "Vision system failed to initialize. Mission aborted.");
            return;
        }

        pointsMapList.put("Area 1", new ArrayList<Pair<Point, Quaternion>>(Arrays.asList(
                new Pair<>(new Point(10.6265d, -9.8406d, 5.00906d), new Quaternion(-0.176166f, -0.176166f, 0.684811f, 0.684811f)),
                new Pair<>(new Point(10.9211d, -9.8406d, 5.00906d), new Quaternion(-0.176166f, -0.176166f, 0.684811f, 0.684811f)),
                new Pair<>(new Point(11.2711d, -9.8406d, 5.00906d), new Quaternion(-0.176166f, -0.176166f, 0.684811f, 0.684811f))
        )));

        pointsMapList.put("Area 2", new ArrayList<Pair<Point, Quaternion>>(Arrays.asList(
                new Pair<>(new Point(11.3432d, -8.92783d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(10.9358d, -8.92783d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(10.5544d, -8.92783d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f))
        )));

        pointsMapList.put("Area 3", new ArrayList<Pair<Point, Quaternion>>(Arrays.asList(
                new Pair<>(new Point(10.5602d, -7.96923d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(10.9503d, -7.96923d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(11.3403d, -7.96923d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f))
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
            String currentImageLabelSuffix = ""; // Declare here to be in scope for saving

            for (int j = 0; j < areaPosList.size(); j++) { // j is imgIndex
                Pair<Point, Quaternion> move = areaPosList.get(j);
                Point point = move.first;
                Quaternion quaternion = move.second;

                Log.d(TAG, "Attempting to move to position " + (j + 1) + " in " + areaName + ": X=" + point.getX() + ", Y=" + point.getY() + ", Z=" + point.getZ());
                try {
                    api.moveTo(point, quaternion, false);
                    Log.d(TAG, "Move command sent to " + areaName + " position " + (j + 1) + ". Check robot logs for actual success.");
                } catch (Exception e) {
                    Log.e(TAG, "Exception during moveTo command for " + areaName + " position " + (j + 1) + ": " + e.getMessage(), e);
                    Log.e(TAG, "Move failed for " + areaName + " pos " + (j + 1) + ": " + e.getMessage());
                    continue;
                }

                Bitmap bitmapDockCam = null;
                Bitmap bitmapNavCam = null;
                Bitmap imageToProcess = null; // ภาพที่จะใช้ในการประมวลผล

                boolean isArea1 = areaName.equals("Area 1");

                try {
                    // Area 1: ใช้ Dock Cam
                    if (isArea1) {
                        Log.d(TAG, "Capturing DockCam for " + areaName + " position " + (j + 1));
                        bitmapDockCam = api.getBitmapDockCam();
                        imageToProcess = bitmapDockCam; // ใช้ Dock Cam ในการประมวลผล
                    } else { // Area 2, 3, 4: ใช้ Nav Cam
                        Log.d(TAG, "Capturing NavCam for " + areaName + " position " + (j + 1));
                        bitmapNavCam = api.getBitmapNavCam();
                        imageToProcess = bitmapNavCam; // ใช้ Nav Cam ในการประมวลผล
                    }

                    List<String> detectedLabelsForCurrentImage = new ArrayList<>();

                    if (imageToProcess != null) {
                        // 1. Get calibrated image (ใช้ภาพที่ได้จาก Dock/Nav Cam)
                        Bitmap preprocessedImageBitmap = bitmapCalibrate(imageToProcess);
                        Log.d(TAG, "Image calibration attempted for " + areaName + " position " + (j + 1));

                        if (preprocessedImageBitmap != null) {
                            // 2. Process image and get detections (ใช้ภาพที่ calibrate แล้ว)
                            List<RobotVisionProcessor.DetectionResult> detections = visionProcessor.processImageAndGetResult(preprocessedImageBitmap);
                            allDetectionsInCurrentArea.addAll(detections);

                            Set<String> uniqueLabelsForImageSet = new HashSet<>();
                            for (RobotVisionProcessor.DetectionResult d : detections) {
                                if (d.confidence >= RobotVisionProcessor.CONFIDENCE_THRESHOLD) {
                                    uniqueLabelsForImageSet.add(d.label);
                                }
                            }
                            List<String> sortedUniqueLabelsForImage = new ArrayList<>(uniqueLabelsForImageSet);
                            Collections.sort(sortedUniqueLabelsForImage);
                            currentImageLabelSuffix = ""; // Reset suffix for potential re-use
                            if (!sortedUniqueLabelsForImage.isEmpty()) {
                                currentImageLabelSuffix = "_" + TextUtils.join("_", sortedUniqueLabelsForImage).replace(" ", "_");
                            } else {
                                currentImageLabelSuffix = "_no_object_detected";
                            }

                            // 3. Save the CALIBRATED image (ไม่ว่าจะเป็น Dock หรือ Nav Cam ที่ calibrate แล้ว)
                            // ให้ระบุว่าเป็น Dock หรือ Nav ในชื่อไฟล์
                            String calibratedFileNamePrefix = isArea1 ? "calibrate_dock_" : "calibrate_nav_";
                            // แก้ไขชื่อตัวแปรตรงนี้ currentImageLablSuffix -> currentImageLabelSuffix
                            api.saveBitmapImage(preprocessedImageBitmap, calibratedFileNamePrefix + i + "_" + j + currentImageLabelSuffix);
                            Log.d(TAG, "Saved CALIBRATED image: " + calibratedFileNamePrefix + i + "_" + j + currentImageLabelSuffix);

                            // 4. Perform the contour-based crop and save
                            Bitmap contourCroppedBitmap = cropImage(preprocessedImageBitmap);
                            if (contourCroppedBitmap != null) {
                                String croppedFileNamePrefix = isArea1 ? "calibrate_crop_dock_" : "calibrate_crop_nav_";
                                String contourCroppedFileName = croppedFileNamePrefix + i + "_" + j + currentImageLabelSuffix + "_contour_new";
                                api.saveBitmapImage(contourCroppedBitmap, contourCroppedFileName);
                                Log.d(TAG, "Saved CALIBRATED+CROPPED image: " + contourCroppedFileName);
                                contourCroppedBitmap.recycle(); // Recycle after saving
                            } else {
                                Log.w(TAG, "Failed to perform contour-based crop for image at " + areaName + " pos " + (j + 1) + ". Cropped image will not be saved.");
                            }

                            detectedLabelsForCurrentImage.addAll(sortedUniqueLabelsForImage);

                            for (RobotVisionProcessor.DetectionResult d : detections) {
                                Log.d(TAG, "Detected in " + areaName + " pos " + (j + 1) + ": " + d.toString());
                            }

                            if (preprocessedImageBitmap != null) {
                                preprocessedImageBitmap.recycle(); // Recycle after all uses
                            }
                        } else {
                            Log.w(TAG, "Calibrated image was null for " + areaName + " position " + (j + 1) + ". Skipping further processing and saving for calibrated/cropped.");
                        }

                    } else {
                        Log.w(TAG, "Camera image (raw) was null at " + areaName + " pos " + (j + 1) + ". No prediction or image saving made.");
                        Log.e(TAG, "Camera image null at " + areaName + " pos " + (j + 1));
                    }

                    // Save the original (raw) images based on the specific area's requirements
                    saveImagePack(bitmapDockCam, bitmapNavCam, i, j, detectedLabelsForCurrentImage, isArea1, !isArea1); // Area 1: save Dock, others: save Nav

                } catch (Exception e) {
                    Log.e(TAG, "Exception during image capture or vision processing for " + areaName + " position " + (j + 1) + ": " + e.getMessage(), e);
                    Log.e(TAG, "Image processing error at " + areaName + " pos " + (j + 1) + ": " + e.getMessage());
                } finally {
                    // Ensure all Bitmaps obtained from API are recycled
                    if (bitmapNavCam != null) {
                        bitmapNavCam.recycle();
                    }
                    if (bitmapDockCam != null) {
                        bitmapDockCam.recycle();
                    }
                    // imageToProcess, preprocessedImageBitmap, contourCroppedBitmap are recycled within the try block if not null
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

                List<String> landmarkItem = new ArrayList<>(Arrays.asList("coin", "compass", "coral", "fossil", "key", "letter", "shell", "treasure_box"));
                List<String> resultItem = new ArrayList<>();
                int resCount = 0;
                Integer areaId = areaNameToIdMap.get(areaName);
                if (areaId != null) {
                    for (String item : uniqueDetectedItems) {
                        for (String lm : landmarkItem) {
                            if (item.equals(lm)) {
                                resultItem.add(item);
                                resCount++;
                            }
                        }
                    }
                    String resultString = TextUtils.join(",", resultItem);
                    api.setAreaInfo(areaId, resultString, resCount);
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
        try {
            bitmapNavCamAstro = api.getBitmapNavCam();

            Bitmap preprocessedNavCamBitmapAstro = null;
            if (bitmapNavCamAstro != null) {
                preprocessedNavCamBitmapAstro = bitmapCalibrate(bitmapNavCamAstro);
                if (preprocessedNavCamBitmapAstro != null) {
                    api.saveBitmapImage(preprocessedNavCamBitmapAstro, "calibrate_astro_999");
                    Log.d(TAG, "Saved CALIBRATED astronaut image: calibrate_astro_999");
                } else {
                    Log.w(TAG, "Calibrated astronaut bitmap is null.");
                }
            } else {
                Log.w(TAG, "Raw astronaut NavCam bitmap is null.");
            }


            Bitmap croppedNavCamBitmapAstro = null;
            if (preprocessedNavCamBitmapAstro != null) { // Checks if calibration was successful
                croppedNavCamBitmapAstro = cropImage(preprocessedNavCamBitmapAstro);
                if (croppedNavCamBitmapAstro != null) {
                    api.saveBitmapImage(croppedNavCamBitmapAstro, "calibrate_crop_astro_999_new");
                    Log.d(TAG, "Saved CALIBRATED+CROPPED astronaut image: calibrate_crop_astro_999_new");
                } else {
                    Log.w(TAG, "Cropped astronaut bitmap is null after cropImage.");
                }
            } else {
                Log.w(TAG, "Preprocessed astronaut bitmap is null, cannot perform crop.");
            }

            List<String> emptyDetectedLabels = new ArrayList<>();
            // Save the original raw astronaut NavCam image only
            saveImagePack(null, bitmapNavCamAstro, -1, 999, emptyDetectedLabels, false, true);

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
        if (distortedImg == null) {
            Log.e(TAG, "bitmapCalibrate: Input distortedImg is null.");
            return null;
        }

        Mat simNavCamMatrix = null;
        Mat simNavCamDistort = null;
        Mat distortedImageMat = null;
        Mat undistortedImageMat = null;
        Bitmap undistortedImageBitmap = null;

        try {
            Double NavCamArr[][] = {
                    {523.105750, 0.000000, 635.434258},
                    {0.000000, 534.765913, 500.335102},
                    {0.000000, 0.000000, 1.000000}
            };
            Double DistCamArr[][] = {{-0.164787, 0.020375, -0.001572, -0.000369, 0.000000}};

            simNavCamMatrix = new Mat(3, 3, CvType.CV_64FC1);
            assignMat(simNavCamMatrix, NavCamArr);
            simNavCamDistort = new Mat(1, 5, CvType.CV_64FC1);
            assignMat(simNavCamDistort, DistCamArr);

            distortedImageMat = new Mat();
            Utils.bitmapToMat(distortedImg, distortedImageMat);
            undistortedImageMat = new Mat();

            Calib3d.undistort(distortedImageMat, undistortedImageMat, simNavCamMatrix, simNavCamDistort);

            undistortedImageBitmap = Bitmap.createBitmap(undistortedImageMat.cols(), undistortedImageMat.rows(), distortedImg.getConfig());
            Utils.matToBitmap(undistortedImageMat, undistortedImageBitmap);

        } catch (Exception e) {
            Log.e(TAG, "Exception in bitmapCalibrate: " + e.getMessage(), e);
            return null; // Return null on error
        } finally {
            if (simNavCamMatrix != null) simNavCamMatrix.release();
            if (simNavCamDistort != null) simNavCamDistort.release();
            if (distortedImageMat != null) distortedImageMat.release();
            if (undistortedImageMat != null) undistortedImageMat.release();
        }
        return undistortedImageBitmap;
    }

    /**
     * Crops an image based on the largest 4-point contour found.
     * Implemented based on the provided Python code.
     *
     * @param inputBitmap The input Bitmap to be cropped.
     * @return The cropped Bitmap, or the original Bitmap if no valid contour is found or cropping fails.
     */
    public static Bitmap cropImage(Bitmap inputBitmap) {
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

        // Sort contours by area
        Collections.sort(contours, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint a, MatOfPoint b) {
                return Double.compare(Imgproc.contourArea(b), Imgproc.contourArea(a));
            }
        });

        // Find a 4-point contour
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        MatOfPoint2f contour2f;
        MatOfPoint location = null;

        for (int i = 0; i < Math.min(contours.size(), 10); i++) {
            contour2f = new MatOfPoint2f(contours.get(i).toArray());
            double epsilon = 0.01 * Imgproc.arcLength(contour2f, true);
            Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true);
            if (approxCurve.total() == 4) {
                location = new MatOfPoint(approxCurve.toArray());
                break;
            }
        }

        if (location == null) {
            Bitmap bitmapImg = Bitmap.createBitmap(img.width(), img.height(), inputBitmap.getConfig());
            Utils.matToBitmap(img, bitmapImg);
            return bitmapImg;
        }

        // Create mask from polygon
        Mat polyMask = Mat.zeros(gray.size(), CvType.CV_8UC1);
        List<MatOfPoint> list = new ArrayList<>();
        list.add(location);
        Imgproc.fillPoly(polyMask, list, new Scalar(255));

        // Apply mask to image
        Mat sampleImage = new Mat();
        Core.bitwise_and(img, img, sampleImage, polyMask);

        // Crop using bounding box of polygon
        int xMin = Integer.MAX_VALUE, xMax = Integer.MIN_VALUE;
        int yMin = Integer.MAX_VALUE, yMax = Integer.MIN_VALUE;

        for (org.opencv.core.Point p : location.toArray()) {
            xMin = Math.min(xMin, (int) p.x);
            xMax = Math.max(xMax, (int) p.x);
            yMin = Math.min(yMin, (int) p.y);
            yMax = Math.max(yMax, (int) p.y);
        }

        Rect roi = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        Bitmap bitmapImg = Bitmap.createBitmap(xMax - xMin, yMax - yMin, inputBitmap.getConfig());
        Utils.matToBitmap(new Mat(sampleImage, roi), bitmapImg);
        return bitmapImg;
    }
}