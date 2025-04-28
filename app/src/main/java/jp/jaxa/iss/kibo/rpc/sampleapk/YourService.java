package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.accessibilityservice.GestureDescription;
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

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    @Override
    protected void runPlan1() {
        // The mission starts.
        api.startMission();

        // Move to a point.
        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        // Get a camera image.
        Mat image = api.getMatNavCam();

        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

        // When you recognize landmark items, let’s set the type and number.

        /* **************************************************** */
        /* Let's move to each area and recognize the items. */

        // Use Map
        Map<String, Pair<Point, Quaternion>> pointsMap = new HashMap<>();
        String patrolPath[] = {"Area 1", "Area 2", "Area 3", "Area 4"};
        List<String> sequencePath = new ArrayList<>(Arrays.asList(patrolPath));

        Map<String, List<Pair<String, String>>> patrolResult = new HashMap<>();  // map[Markers] = list( pair<string, string> )

        pointsMap.put("Area 1", new Pair<>(new Point(10.8d, -9.7d, 4.7d), new Quaternion(0.687483f, -0.164974f, 0.164974f, 0.687483f)));
        if (pointsMap.isEmpty()) {
            System.out.println("ERROR: 'pointsMap' is empty");
            api.shutdownFactory();
        }
        pointsMap.put("Area 2", new Pair<>(new Point(10.9d, -8.8d, 4.6d), new Quaternion(0.5f, -0.5f, 0.5f, 0.5f)));
        pointsMap.put("Area 3", new Pair<>(new Point(10.9d, -7.9d, 4.5d), new Quaternion(0.5f, -0.5f, 0.5f, 0.5f)));
        pointsMap.put("Area 4", new Pair<>(new Point(10.6d, -6.7d, 5.0d), new Quaternion(0f, 0f, 0f, 1f)));
        pointsMap.put("Area 5", new Pair<>(new Point(11.1d, -6.9d, 4.8d), new Quaternion(0f, 0f, -0.707f, 0.707f)));

        /*
        for (Map.Entry<String, Pair<Point, Quaternion>> entry : pointsMap.entrySet()) {*/
        for (int i=0; i < sequencePath.size(); i++) {
            String area = sequencePath.get(i);
            Pair<Point, Quaternion> pair = pointsMap.get(area);

            // Move through every area
            point = pair.first;
            quaternion = pair.second;
            api.moveTo(point, quaternion, true);

            // Capture markers and treasures
            int itemQuant = 2; // Item Quantity
            String itemName = "itemName"; // Item Name

            api.setAreaInfo(i + 1, area, itemQuant);
        }


        // When you move to the front of the astronaut, report the rounding completion.
        point = new Point(11.143d, -6.7607d, 4.9654d);
        quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point, quaternion, false);
        api.reportRoundingCompletion();

        /* ********************************************************** */
        /* Write your code to recognize which target item the astronaut has. */
        String target = new String("crystal");
        Pair<String, String> astroMarker = new Pair<>("coin", "compass"); // First and Second Target landmark
        String targetArea = "Area 1";

        if (patrolResult.containsKey(astroMarker.first)) {
            List<Pair<String, String>> patrolList = patrolResult.get(astroMarker.first);
            for (int i=0; i < patrolList.size(); i++) {
                String treasure = patrolList.get(i).first;
                if (target == treasure) {
                    targetArea = patrolList.get(i).second;
                }
            }
        } else if (patrolResult.containsKey(astroMarker.second)) {
            List<Pair<String, String>> patrolList = patrolResult.get(astroMarker.second);
            for (int i=0; i < patrolList.size(); i++) {
                String treasure = patrolList.get(i).first;
                if (target == treasure) {
                    targetArea = patrolList.get(i).second;
                }
            }
        } else {
            System.out.print("No Marker Detected");
        }
        // If target is Area 2:


        /* ********************************************************** */

        // Let's notify the astronaut when you recognize it.
        api.notifyRecognitionItem();



        Pair<Point, Quaternion> targetCoordinate = new Pair<>(pointsMap.get(targetArea).first, pointsMap.get(targetArea).second);
        api.moveTo(targetCoordinate.first, targetCoordinate.second, false); // go to target area
        /* ******************************************************************************************************* */
        /* Write your code to move Astrobee to the location of the target item (what the astronaut is looking for) */
        /* ******************************************************************************************************* */

        // Take a snapshot of the target item.
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