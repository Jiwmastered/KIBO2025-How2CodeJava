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
import java.util.Vector;

import javax.crypto.NullCipher;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    private Map<String, Integer> itemTypeMap = new HashMap<>(); // Map of item name and type, 1=landmark, 2=treasure
    private Map<String, List<Pair<String, String>>> patrolResultMap = new HashMap<>();  // map[Markers] = list( pair<string, string> )
    private Map<String, Pair<Point, Quaternion>> pointsMap = new HashMap<>();

    private void saveImagePack(Bitmap bitmapDockCam, Mat matDockCam, Bitmap bitmapNavCam, Mat matNavCam, int areaId) {
        api.saveBitmapImage(bitmapDockCam, "bit_dock_area_" + (areaId+1));
        api.saveMatImage(matDockCam, "mat_dock_area_"+ (areaId+1));
        api.saveBitmapImage(bitmapNavCam, "bit_nav_area_" + (areaId+1));
        api.saveMatImage(matNavCam, "mat_nav_area_"+ (areaId+1));
    }


    @Override
    protected void runPlan1() {
        // The mission starts.
        api.startMission();

        // Move to a point.
        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        // Rotation Test
        System.out.println("StartRotating");
        api.moveTo(point, new Quaternion(1f, 0f, 0f, 0f), false);
        api.moveTo(point, new Quaternion(0f, 1f, 0f, 0f), false);
        api.moveTo(point, new Quaternion(0f, 0f, 1f, 0f), false);
        api.moveTo(point, new Quaternion(0f, 0f, 0f, 1f), false);


        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

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

        pointsMap.put("Area 1", new Pair<>(new Point(10.8d, -9.7d, 4.7d), new Quaternion(0.687483f, -0.164974f, 0.164974f, 0.687483f)));
        if (pointsMap.isEmpty()) {
            System.out.println("ERROR: 'pointsMap' is empty");
            api.shutdownFactory();
        }
        pointsMap.put("Area 2", new Pair<>(new Point(10.9d, -8.8d, 4.6d), new Quaternion(0.5f, -0.5f, 0.5f, 0.5f)));
        pointsMap.put("Area 3", new Pair<>(new Point(10.9d, -7.9d, 4.5d), new Quaternion(0.5f, -0.5f, 0.5f, 0.5f)));
        pointsMap.put("Area 4", new Pair<>(new Point(10.6d, -6.7d, 5.0d), new Quaternion(0f, 0f, 0f, 1f)));
        pointsMap.put("Area 5", new Pair<>(new Point(11.1d, -6.9d, 4.8d), new Quaternion(0f, 0f, -0.707f, 0.707f)));

        String patrolPath[] = {"Area 1", "Area 2", "Area 3", "Area 4"};
        List<String> sequencePath = new ArrayList<>(Arrays.asList(patrolPath));

        for (int i=0; i < sequencePath.size(); i++) {
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
            saveImagePack(bitmapDockCam, matDockCam, bitmapNavCam, matNavCam, i);
            
            /*
            objectIdenification() --> get list<string> of detected object, detection time = 1000ms,
            List<String> detectionResult = objectList;
            for (int it=0; it<detectionResult.size(); it++) {
                String item = detectionResult.get(it);
                if (itemTypeMap.get(item) == 1) {
                    itemName = item;
                    itemQuantity++;
                } else {
                    treasure = item;
                }
            }
            */

            // Detected object variable
            String itemName = "itemName";
            String treasure = "None";
            int itemQuantity = 0;

            // Check if marker is mapped, append pair of treasure name and area
            Pair<String, String> markerDat = Pair.create(treasure, areaName);
            if (!patrolResultMap.containsKey(itemName)){
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

        // When you move to the front of the astronaut, report the rounding completion.
        point = new Point(11.143d, -6.7607d, 4.9654d);
        quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point, quaternion, false);
        api.reportRoundingCompletion();
        //          save img
        Bitmap bitmapNavCam = api.getBitmapNavCam();
        Mat matNavCam = api.getMatNavCam();
        api.saveBitmapImage(bitmapNavCam, "bit_nav_astro");
        api.saveMatImage(matNavCam, "mat_nav_astro");


        /* ********************************************************** */
        /* Write your code to recognize which target item the astronaut has. */

        String targetItem = new String("crystal");
        Pair<String, String> astroMarker = new Pair<>("coin", "compass"); // First and Second Target landmark
        String targetArea = "Area 1";

        if (patrolResultMap.containsKey(astroMarker.first)) {
            List<Pair<String, String>> patrolList = patrolResultMap.get(astroMarker.first);
            for (int i=0; i < patrolList.size(); i++) {
                String treasure = patrolList.get(i).first;
                if (targetItem.equals(treasure)) {
                    targetArea = patrolList.get(i).second;
                    System.out.print("Marker Detected at " + targetArea);
                }
            }
        } else if (patrolResultMap.containsKey(astroMarker.second)) {
            List<Pair<String, String>> patrolList = patrolResultMap.get(astroMarker.second);
            for (int i=0; i < patrolList.size(); i++) {
                String treasure = patrolList.get(i).first;
                if (targetItem.equals(treasure)) {
                    targetArea = patrolList.get(i).second;
                    System.out.print("Marker Detected at " + targetArea);
                }
            }
        } else {
            System.out.print("No Marker Detected");
        }

        /* ********************************************************** */
        // Let's notify the astronaut when you recognize it.
        api.notifyRecognitionItem();


        // Move to target area and Take a snapshot of the target item.
        Pair<Point, Quaternion> targetCoordinate = new Pair<>(pointsMap.get(targetArea).first, pointsMap.get(targetArea).second);
        api.moveTo(targetCoordinate.first, targetCoordinate.second, false); // go to target area
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