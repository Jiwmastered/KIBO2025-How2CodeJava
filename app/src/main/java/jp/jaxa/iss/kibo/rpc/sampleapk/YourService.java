package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.accessibilityservice.GestureDescription;
import android.graphics.Bitmap;
import android.util.Pair;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.core.Mat;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.Vector;

import javax.crypto.NullCipher;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    private Map<String, Integer> itemTypeMap = new HashMap<>(); // Map of item name and type, 1=landmark, 2=treasure // May no required
    private Map<String, List<Pair<String, String>>> patrolResultMap = new HashMap<>();  // map[Markers] = list( pair<string, string> )
    private Map<String, Pair<Point, Quaternion>> pointsMap = new HashMap<>();
    private Map<String, List<Pair>> pointsMapList = new HashMap<>();
    private Map<String, Pair<String, String>> areaInfoMap = new HashMap<>(); // Map<Landmark, Pair<Area, Treasure>>
    private Pair itemTarget = Pair.create("", ""); // Pair<Landmark, Treasure>

    private List<String> treasureList = Arrays.asList("crystal", "diamond", "emerald");
    private List<String> landmarkList = Arrays.asList("coin", "compass", "coral", "fossil", "key", "letter", "shell", "treasure_box");

    private void saveImagePack(Bitmap bitmapDockCam, Mat matDockCam, Bitmap bitmapNavCam, Mat matNavCam, int areaId, int imgId) {
        api.saveBitmapImage(bitmapDockCam, "bit_dock_area_" + (areaId) + "_" + imgId);
        api.saveMatImage(matDockCam, "mat_dock_area_" + (areaId) + "_" + imgId);
        api.saveBitmapImage(bitmapNavCam, "bit_nav_area_" + (areaId) + "_" + imgId);
        api.saveMatImage(matNavCam, "mat_nav_area_" + (areaId) + "_" + imgId);
    }

    //      Map<LMItem, Pair<Area, Treasure>>
    private Map<String, Pair<String, String>> imageRecognition(int area /* IDK WHAT TO INPUT */) {

        Map<String, Pair<String, String>> areaInfoMap = new HashMap<>();

        // TODO: Use API here
        areaInfoMap.put("Area " + area, Pair.create("use api recog", "use api recog"));
        return areaInfoMap;
    }

    @Override
    protected void runPlan1() {
        // The mission starts.
        api.startMission();

        // Move to a point.

        // Initiate Point and Quaternion
        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);

        // May no required
        itemTypeMap.put("coin", 1);
        itemTypeMap.put("compass", 1);
        itemTypeMap.put("coral", 1);
        itemTypeMap.put("fossil", 1);
        itemTypeMap.put("key", 1);
        itemTypeMap.put("letter", 1);
        itemTypeMap.put("shell", 1);
        itemTypeMap.put("treasure_box", 1);
        itemTypeMap.put("crystal", 2);
        itemTypeMap.put("diamond", 2);
        itemTypeMap.put("emerald", 2);


        // TODO: Second move not set
        pointsMapList.put("Test 1", new ArrayList<Pair>(Arrays.asList(
                new Pair<>(new Point(10.8d, -9.7d, 4.7d), new Quaternion(-0.231908f, -0.231908f, -0.667883f, 0.667883f)),
                new Pair<>(new Point(10.8d, -9.7d, 4.7d), new Quaternion(-0.231908f, -0.231908f, -0.667883f, 0.667883f))
        )));

        pointsMapList.put("Test 2", new ArrayList<Pair>(Arrays.asList(
                new Pair<>(new Point(10.9d, -8.8d, 4.65d), new Quaternion(-0.5f, 0.5f, 0.5f, 0.5f)),
                new Pair<>(new Point(10.8d, -9.7d, 4.7d), new Quaternion(-0.231908f, -0.231908f, -0.667883f, 0.667883f))
        )));

        pointsMapList.put("Test 3", new ArrayList<Pair>(Arrays.asList(
                new Pair<>(new Point(10.9d, -7.9d, 4.65d), new Quaternion(-0.5f, 0.5f, 0.5f, 0.5f)),
                new Pair<>(new Point(10.8d, -9.7d, 4.7d), new Quaternion(-0.231908f, -0.231908f, -0.667883f, 0.667883f))
        )));

        pointsMapList.put("Test 4", new ArrayList<Pair>(Arrays.asList(
                new Pair<>(new Point(10.6d, -6.7d, 5.0d), new Quaternion(0f, 0f, 1f, 0f)),
                new Pair<>(new Point(10.8d, -9.7d, 4.7d), new Quaternion(-0.231908f, -0.231908f, -0.667883f, 0.667883f))
        )));

        pointsMapList.put("Test 5", new ArrayList<Pair>(Arrays.asList(
                new Pair<>(new Point(11.1d, -6.9d, 4.8d), new Quaternion(0f, 0f, -0.707f, 0.707f)),
                new Pair<>(new Point(10.8d, -9.7d, 4.7d), new Quaternion(-0.231908f, -0.231908f, -0.667883f, 0.667883f))
        )));


        pointsMap.put("Area 1", new Pair<>(new Point(10.8d, -9.7d, 4.7d), new Quaternion(-0.231908f, -0.231908f, -0.667883f, 0.667883f)));
        pointsMap.put("Area 2", new Pair<>(new Point(10.9d, -8.8d, 4.65d), new Quaternion(-0.5f, 0.5f, 0.5f, 0.5f)));
        pointsMap.put("Area 3", new Pair<>(new Point(10.9d, -7.9d, 4.65d), new Quaternion(-0.5f, 0.5f, 0.5f, 0.5f)));
        pointsMap.put("Area 4", new Pair<>(new Point(10.6d, -6.7d, 5.0d), new Quaternion(0f, 0f, 1f, 0f)));

        pointsMap.put("Area 5", new Pair<>(new Point(11.1d, -6.9d, 4.8d), new Quaternion(0f, 0f, -0.707f, 0.707f)));

        String[] patrolPath = {"Area 1", "Area 2", "Area 3", "Area 4"};
        List<String> sequencePath = new ArrayList<>(Arrays.asList(patrolPath));


//        Map<String, List<Pair>> pointsMapList = new HashMap<>();
//        private Map<String, Pair<String, String>> areaInfoMap = new HashMap<>(); // Map<Landmark, Pair<Area, Treasure>>
//        private Pair itemTarget = Pair.create("", "");

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
                Bitmap bitmapDockCam = api.getBitmapDockCam();
                Mat matDockCam = api.getMatDockCam();
                Bitmap bitmapNavCam = api.getBitmapNavCam();
                Mat matNavCam = api.getMatNavCam();
                saveImagePack(bitmapDockCam, matDockCam, bitmapNavCam, matNavCam, i, j);

            }

            // Recognition Operation
            // Assume: Result Area 2
            areaInfoMap = imageRecognition(i + 1 /* And some Image stuff */);

            // Handing Treasure not exist (If be null)
//            boolean hasTreasure;
//            for (String treasure : treasureList) {
//                if (treasure == areaInfoMap.)
//            }

        }


        // Jiw's Code
/*
        for (int i = 0; i < sequencePath.size(); i++) {
            String areaName = sequencePath.get(i);
            Pair<Point, Quaternion> areaPosition = pointsMap.get(areaName);

            // Move through every area
            point = areaPosition.first;
            quaternion = areaPosition.second;
            api.moveTo(point, quaternion, true);

            // Capture markers and treasures, Save img
            Bitmap bitmapDockCam = api.getBitmapDockCam();
            Mat matDockCam = api.getMatDockCam();
            Bitmap bitmapNavCam = api.getBitmapNavCam();
            Mat matNavCam = api.getMatNavCam();
            saveImagePack(bitmapDockCam, matDockCam, bitmapNavCam, matNavCam, i, 1);


//            objectIdenification() --> get list<string> of detected object, detection time = 1000ms,
//            List<String> detectionResult = objectList;
//            for (int it=0; it<detectionResult.size(); it++) {
//                String item = detectionResult.get(it);
//                if (itemTypeMap.get(item) == 1) {
//                    itemName = item;
//                    itemQuantity++;
//                } else {
//                    treasure = item;
//                }
//            }


            // Detected object variable
            String itemName = "itemName";
            String treasure = "None";
            int itemQuantity = 0;

            // Check if marker is mapped, append pair of treasure name and area
            Pair<String, String> markerDat = Pair.create(treasure, areaName);
            if (!patrolResultMap.containsKey(itemName)) {
                List<Pair<String, String>> markerList = new ArrayList<>(Arrays.asList(markerDat));
                patrolResultMap.put(itemName, markerList);
                //System.out.println(markerList.get(0).first);
            } else {
                List<Pair<String, String>> markerList = patrolResultMap.get(itemName);
                markerList.add(markerDat);
                patrolResultMap.put(itemName, markerList);
            }

            api.setAreaInfo(i + 1, itemName, itemQuantity);
        }
        */

        // When you move to the front of the astronaut, report the rounding completion.
        point = new Point(11.143d, -6.7607d, 4.9654d);
        quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        //          save img
        Bitmap bitmapNavCam = api.getBitmapNavCam();
        Mat matNavCam = api.getMatNavCam();
        api.saveBitmapImage(bitmapNavCam, "bit_nav_astro");
        api.saveMatImage(matNavCam, "mat_nav_astro");

        api.reportRoundingCompletion();

        // TODO: Use API here
        itemTarget = Pair.create("use api recognizing", "use api recognizing");

//        private List<String> treasureList = Arrays.asList("crystal", "diamond", "emerald");
//        private List<String> landmarkList = Arrays.asList("coin", "compass", "coral", "fossil", "key", "letter", "shell", "treasure_box");

        String areaTarget = "";

        for (Map.Entry<String, Pair<String, String>> entry : areaInfoMap.entrySet()) {
            String landmark = entry.getKey();
            Pair<String, String> areaTreasurePair = entry.getValue();
            String area = areaTreasurePair.first;
            String treasure = areaTreasurePair.second;

            // Check landmark and treasure are matched
            if (landmark.equals(itemTarget.first) && treasure.equals(itemTarget.second)) {
                areaTarget = area;
            }
        }

        api.notifyRecognitionItem();

        // Get ready to go to Target
        List<Pair> targetCoordList = pointsMapList.get(areaTarget);
        Pair<Point, Quaternion> targetCoordPair = targetCoordList.get(0);
        point = targetCoordPair.first;
        quaternion = targetCoordPair.second;

        api.moveTo(point, quaternion, false);

        // Jiw's Code

//        String targetItem = new String("crystal");
//        Pair<String, String> astroMarker = new Pair<>("coin", "compass"); // First and Second Target landmark
//        String targetArea = "Area 1";
//
//        if (patrolResultMap.containsKey(astroMarker.first)) {
//            List<Pair<String, String>> patrolList = patrolResultMap.get(astroMarker.first);
//            for (int i = 0; i < patrolList.size(); i++) {
//                String treasure = patrolList.get(i).first;
//                if (targetItem.equals(treasure)) {
//                    targetArea = patrolList.get(i).second;
//                    System.out.print("Marker Detected at " + targetArea);
//                }
//            }
//        } else if (patrolResultMap.containsKey(astroMarker.second)) {
//            List<Pair<String, String>> patrolList = patrolResultMap.get(astroMarker.second);
//            for (int i = 0; i < patrolList.size(); i++) {
//                String treasure = patrolList.get(i).first;
//                if (targetItem.equals(treasure)) {
//                    targetArea = patrolList.get(i).second;
//                    System.out.print("Marker Detected at " + targetArea);
//                }
//            }
//        } else {
//            System.out.print("No Marker Detected");
//        }

        /* ********************************************************** */
        // Let's notify the astronaut when you recognize it.

//        api.notifyRecognitionItem();


        // Move to target area and Take a snapshot of the target item.
//        Pair<Point, Quaternion> targetCoordinate = new Pair<>(pointsMap.get(targetArea).first, pointsMap.get(targetArea).second);
//        api.moveTo(targetCoordinate.first, targetCoordinate.second, false); // go to target area


        // Finish
        api.takeTargetItemSnapshot();
    }

    // No use
    /*
    @Override
    protected void runPlan2(){
       // write your plan 2 here.
        api.startMission();

        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        point = new Point(11.143d, -6.7607d, 4.9654d);
        quaternion = new Quaternion(0, 0, 0.5f, 0.5f);

    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }
    */

    // You can add your method.
    private String yourMethod() {
        return "your method";
    }
}