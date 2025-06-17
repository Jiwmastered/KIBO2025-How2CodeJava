package jp.jaxa.iss.kibo.rpc.sampleapk;



import android.accessibilityservice.GestureDescription;
import android.graphics.Bitmap;
import android.util.Pair;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.android.Utils;

import org.opencv.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CamUtils extends KiboRpcService {
    private static void assignMat(Mat mat, Double arr[][]){
        for (int i=0; i<arr.length; i++) {
            for (int j=0; j<arr[i].length; j++) {
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

        int imageWidth = 1280;
        int imageHeight = 960;

        // distort
        Mat distortedImageMat = new Mat();
        Utils.bitmapToMat(distortedImg, distortedImageMat);
        Mat undistortedImageMat = new Mat();
        Bitmap undistortedImageBitmap = Bitmap.createBitmap(imageWidth, imageHeight, distortedImg.getConfig());

        // undistort
        Calib3d.undistort(distortedImageMat, undistortedImageMat, simNavCamMatrix, simNavCamDistort);

        Utils.matToBitmap(undistortedImageMat, undistortedImageBitmap);
        return undistortedImageBitmap;
    }

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


    public static Map<List<String>, Pair<String, String>> imageRecognition(int i) {
        Map<List<String>, Pair<String, String>> areaInfoMap = new HashMap<>();
        List<String> landmarks = new ArrayList<>();

        // TODO: Use API here


        areaInfoMap.put(landmarks, Pair.create("area", "treasure"));
        return areaInfoMap;
    }
}