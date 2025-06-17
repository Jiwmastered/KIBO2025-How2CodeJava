package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.accessibilityservice.GestureDescription;
import android.graphics.Bitmap;
import android.util.Pair;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Arrays;

import javax.crypto.NullCipher;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
//    private Map<String, Integer> itemTypeMap = new HashMap<>(); // Map of item name and type, 1=landmark, 2=treasure // May no required
//    private List<String> treasureList = Arrays.asList("crystal", "diamond", "emerald");
//    private List<String> landmarkList = Arrays.asList("coin", "compass", "coral", "fossil", "key", "letter", "shell", "treasure_box");
    ObjectDetectionJar ObjDetectionSession = new ObjectDetectionJar();

    private static Map<String, List<Pair>> pointMapList() {
        Map<String, List<Pair>> pMapList = new HashMap<>();
        pMapList.put("Area 1", new ArrayList<Pair>(Arrays.asList(
                new Pair<>(new Point(10.6265d, -10.0406d, 4.75906d), new Quaternion(-0.011651f, -0.011651f, -0.706904f, 0.706904f)),
                new Pair<>(new Point(10.9211d, -10.0406d, 4.75906d), new Quaternion(-0.011651f, -0.011651f, -0.706904f, 0.706904f)),
                new Pair<>(new Point(11.2711d, -10.0406d, 4.75906d), new Quaternion(-0.011651f, -0.011651f, -0.706904f, 0.706904f)),

                // Extend views for image capturing
                new Pair<>(new Point(10.9211d, -10.0348d, 5.20394d), new Quaternion(-0.011651f, -0.011651f, -0.706904f, 0.706904f)),
                new Pair<>(new Point(10.9211d, -10.0348d, 5.20394d), new Quaternion(-0.008703f, -0.013991f, -0.528058f, 0.848871f)),
                new Pair<>(new Point(10.9211d, -10.0348d, 5.20394d), new Quaternion(-0.013303f, -0.009722f, -0.807151f, 0.58986f)),
                new Pair<>(new Point(10.9211d, -10.0348d, 5.20394d), new Quaternion(0.679825f, -0.73313f, -0.002777f, 0.007045f)),
                new Pair<>(new Point(10.9211d, -10.0348d, 5.20394d), new Quaternion(0.895047f, -0.445569f, -0.000108f, 0.007572f)),
                new Pair<>(new Point(10.9211d, -10.0348d, 5.20394d), new Quaternion(0.635487f, -0.771879f, -0.003187f, 0.00687f))
        )));

        pMapList.put("Area 2", new ArrayList<Pair>(Arrays.asList(
                new Pair<>(new Point(11.3432d, -8.92783d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(10.9358d, -8.92783d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                // Extend views for image capturing
                new Pair<>(new Point(10.9358d, -8.92783d, 4.45397d), new Quaternion(0.127238f, 0.695565f, 0.127238f, 0.695565f)),
                new Pair<>(new Point(10.9358d, -8.92783d, 4.45397d), new Quaternion(-0.081032f, 0.702449f, -0.081032f, 0.702449f)),

                new Pair<>(new Point(10.5544d, -8.92783d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f))
        )));

        pMapList.put("Area 3", new ArrayList<Pair>(Arrays.asList(
                new Pair<>(new Point(10.5602d, -7.96923d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                new Pair<>(new Point(10.9503d, -7.96923d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f)),
                // Extend views for image capturing
                new Pair<>(new Point(10.9503d, -7.96923d, 4.45397d), new Quaternion(0.127238f, 0.695565f, 0.127238f, 0.695565f)),
                new Pair<>(new Point(10.9503d, -7.96923d, 4.45397d), new Quaternion(-0.081032f, 0.702449f, -0.081032f, 0.702449f)),

                new Pair<>(new Point(11.3403d, -7.96923d, 4.45397d), new Quaternion(0f, 0.707107f, 0f, 0.707107f))
        )));

        pMapList.put("Area 4", new ArrayList<Pair>(Arrays.asList(
                new Pair<>(new Point(10.6149d, -6.82423d, 4.55599d), new Quaternion(0f, 0f, 1f, 0f)),
                new Pair<>(new Point(10.6149d, -6.82423d, 4.9114d), new Quaternion(0f, 0f, 1f, 0f)),
                new Pair<>(new Point(10.6149d, -6.82423d, 5.31016d), new Quaternion(0f, 0f, 1f, 0f))
        )));
        return pMapList;
    }


    public void saveImagePack(Bitmap bitmapDockCam, Mat matDockCam, Bitmap bitmapNavCam, Mat matNavCam, int areaId, int imgId) {
        //api.saveBitmapImage(bitmapDockCam, "bit_dock_area_" + areaId + "_" + imgId + ".png");
        //api.saveMatImage(matDockCam, "mat_dock_area_" + areaId + "_" + imgId + ".png");
        api.saveBitmapImage(bitmapNavCam, "bit_nav_area_" + areaId + "_" + imgId + ".png");
        //api.saveMatImage(matNavCam, "mat_nav_area_" + areaId + "_" + imgId + ".mat");
    }

    @Override
    protected void runPlan1() {
        // Deprecated Map<String, List<Pair<String, String>>> patrolResultMap = new HashMap<>();  // map[Markers] = list( pair<string, string> )
        // Deprecated Map<String, Pair<Point, Quaternion>> pointsMap = new HashMap<>();

        Map<String, List<Pair>> pointsMapList = pointMapList();
        Map<List<String>, Pair<String, String>> areaInfoMap = new HashMap<>(); // Map<Landmark, Pair<Area, Treasure>>
        //ObjDetectionSession.createInterpreter(this);

        /*
        | PATROLLING PHASE |
        */

        api.startMission(); // Start mission
        Point point = new Point();
        Quaternion quaternion = new Quaternion();

        List<String> sequencePath = new ArrayList<>(Arrays.asList("Area 1", "Area 2", "Area 3", "Area 4"));

        for (int i = 0; i < sequencePath.size(); i++) {
            String areaName = sequencePath.get(i);
            List<Pair> areaPosList = pointsMapList.get(areaName);

            // Move to Area
            for (int j = 0; j < areaPosList.size(); j++) {

                // Move to each Position in Area
                Pair<Point, Quaternion> move = areaPosList.get(j);
                point = move.first;
                quaternion = move.second;
                api.moveTo(point, quaternion, false);

                // Capture markers and treasures, Save img
                Bitmap bitDock = api.getBitmapDockCam();
                Bitmap bitNav = api.getBitmapNavCam();
                Bitmap refinedNav = CamUtils.cropImage(CamUtils.bitmapCalibrate(bitNav));
                saveImagePack(bitDock, null, refinedNav, null, i, j);
                /*
                ArrayList<ArrayList<ArrayList<Float>>> result = ObjDetectionSession.runInference(bitmapNavCam);
                System.out.println("Result 1 \n");
                for (ArrayList<ArrayList<Float>> result1 : result) {
                    for (ArrayList<Float> result2 : result1) {
                        for (Float ele : result2) {
                            System.out.println(ele + " ");
                        }
                        System.out.println("\n");
                    }
                    System.out.println("\n");
                }
                System.out.println("\n");

                */
            }
            // Recognition Operation
            // Assume: Result Area 2

            areaInfoMap = CamUtils.imageRecognition(i + 1 /* And some Image stuff */);
        }

        /*
        | RECOGNIZE TARGET |
        */

        // When you move to the front of the astronaut, report the rounding completion.
        point = new Point(11.143d, -6.7607d, 4.9654d);
        quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        // Save image
        api.reportRoundingCompletion();
        Bitmap bitmapNavCam = api.getBitmapNavCam();
        Mat matNavCam = api.getMatNavCam();
        Mat testMat = new Mat();

        api.saveBitmapImage(bitmapNavCam, "bit_nav_astro.png");
        api.saveMatImage(matNavCam, "mat_nav_astro.mat");

        // TODO: Recognize astronaut's deed
        String landmark1Astro = "landmark1";
        String landmark2Astro = "landmark2";
        String treasureAstro = "treasure1";
        String areaTarget = "";

        Pair itemTarget = Pair.create(Pair.create(landmark1Astro, landmark2Astro), treasureAstro);
        // Loop through every areaInfoMap
        for (Map.Entry<List<String>, Pair<String, String>> entry : areaInfoMap.entrySet()) {
            List<String> landmarks = entry.getKey();
            Pair<String, String> areaTreasurePair = entry.getValue();

            String area = areaTreasurePair.first;
            String treasure = areaTreasurePair.second;
            // Check landmark and treasure are matched
            for (String landmark : landmarks) {
                if (landmark.equals(itemTarget.first) && treasure.equals(itemTarget.second)) {
                    areaTarget = area;
                }
            }
        }

        api.notifyRecognitionItem();

        // Get ready to go to Target
        List<Pair> targetCordList = pointsMapList.get(areaTarget);
        Pair<Point, Quaternion> targetCordPair = targetCordList.get(0);
        point = targetCordPair.first;
        quaternion = targetCordPair.second;

        api.moveTo(point, quaternion, false);
        api.takeTargetItemSnapshot();
    }
}