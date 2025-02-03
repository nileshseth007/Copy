import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ImageProcessor {

    public static List<Mat> calculateOpticalFlow(List<Mat> images, String method, boolean fromCache, String flowmapDir) {
        if (flowmapDir != null) {
            flowmapDir = flowmapDir + "/" + method;
            new File(flowmapDir).mkdirs();
        }

        List<Mat> flowMaps = new ArrayList<>();

        // Load from cache if available
        if (fromCache && flowmapDir != null && new File(flowmapDir).exists()) {
            File[] files = new File(flowmapDir).listFiles((dir, name) -> name.endsWith(".npy"));
            if (files != null && files.length > 0) {
                for (File file : files) {
                    Mat flowmap = ImageUtils.loadNpyAsMat(file.getAbsolutePath());
                    flowMaps.add(flowmap);
                }
                return flowMaps;
            }
        }

        if (method.equalsIgnoreCase("cv2")) {
            // Convert images to grayscale
            List<Mat> grayImages = new ArrayList<>();
            for (Mat img : images) {
                Mat gray = new Mat();
                Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
                grayImages.add(gray);
            }

            // Calculate pair-wise optical flow maps using Farneback method
            for (int i = 1; i < grayImages.size(); i++) {
                Mat flow = new Mat();
                Video.calcOpticalFlowFarneback(grayImages.get(i - 1), grayImages.get(i), flow, 
                                               0.5, 5, 11, 5, 5, 1.1, 0);
                flowMaps.add(flow);
            }
        } else if (method.equalsIgnoreCase("raft")) {
            // Calculate pair-wise optical flow maps using RAFT (DIS Optical Flow in Java)
            flowMaps = Raft.calculateRaftOpticalFlow(images);
        }

        // Cache results if possible
        if (flowmapDir != null) {
            for (int i = 0; i < flowMaps.size(); i++) {
                ImageUtils.saveMatAsNpy(flowmapDir + "/" + method + "_flowmap_" + String.format("%03d", i), flowMaps.get(i));
            }
        }

        return flowMaps;
    }
}
public static Pair<Mat, Mat> composite(Mat sharpImage, Mat blurredImage, Mat[] flowMaps, Mat subjectMask) {
    // Compute MFlow
    Mat MFlow = Composite.calcMFlow(flowMaps, sharpImage);

    // Normalize MFlow
    Mat normalizedMFlow = new Mat();
    Core.normalize(MFlow, normalizedMFlow, 0, 1, Core.NORM_MINMAX);

    // Combine the flow and the clipped face masks with a max operator
    Mat flowFaceMask = new Mat();
    Core.max(normalizedMFlow, subjectMask, flowFaceMask);

    // Normalize flowFaceMask
    Core.normalize(flowFaceMask, flowFaceMask, 0, 1, Core.NORM_MINMAX);

    // Perform alpha blending
    Mat compositeImage = Composite.alphaBlending(sharpImage, flowFaceMask, blurredImage);

    return new Pair<>(compositeImage, flowFaceMask);
}
/*
// in the pipeline

Pair<Mat, Mat> compositeResult = ImageProcessor.composite(sharpImage, blurredImage, flowMaps, subjectMask);
Mat resultImage = compositeResult.first;
Mat flowFaceMask = compositeResult.second;

// Save the results
Imgcodecs.imwrite(outputDirectory + "/result.png", resultImage);
Imgcodecs.imwrite(outputDirectory + "/flow_face_mask.png", flowFaceMask);
*/

// calculateOpticalFlow
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ImageProcessor {

    public static List<Mat> calculateOpticalFlow(List<Mat> images, String method, boolean fromCache, String flowmapDir) {
        if (flowmapDir != null) {
            flowmapDir = flowmapDir + "/" + method;
            new File(flowmapDir).mkdirs();
        }

        List<Mat> flowMaps = new ArrayList<>();

        // Load from cache if available
        if (fromCache && flowmapDir != null && new File(flowmapDir).exists()) {
            File[] files = new File(flowmapDir).listFiles((dir, name) -> name.endsWith(".npy"));
            if (files != null && files.length > 0) {
                for (File file : files) {
                    Mat flowmap = ImageUtils.loadNpyAsMat(file.getAbsolutePath());
                    flowMaps.add(flowmap);
                }
                return flowMaps;
            }
        }

        if (method.equalsIgnoreCase("cv2")) {
            // Convert images to grayscale
            List<Mat> grayImages = new ArrayList<>();
            for (Mat img : images) {
                Mat gray = new Mat();
                Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
                grayImages.add(gray);
            }

            // Calculate pair-wise optical flow maps using Farneback method
            for (int i = 1; i < grayImages.size(); i++) {
                Mat flow = new Mat();
                Video.calcOpticalFlowFarneback(grayImages.get(i - 1), grayImages.get(i), flow, 
                                               0.5, 5, 11, 5, 5, 1.1, 0);
                flowMaps.add(flow);
            }
        } else if (method.equalsIgnoreCase("raft")) {
            // Calculate pair-wise optical flow maps using RAFT (DIS Optical Flow in Java)
            flowMaps = Raft.calculateRaftOpticalFlow(images);
        }

        // Cache results if possible
        if (flowmapDir != null) {
            for (int i = 0; i < flowMaps.size(); i++) {
                ImageUtils.saveMatAsNpy(flowmapDir + "/" + method + "_flowmap_" + String.format("%03d", i), flowMaps.get(i));
            }
        }

        return flowMaps;
    }
}

// update in pipeline function
// List<Mat> flowMaps = ImageProcessor.calculateOpticalFlow(images, method, false, flowmapDirectory);

// for getMask
//Mat subjectMask = DrawMask.getMask(sharpImage);

