import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.CvType;
import org.opencv.core.Point;
import org.opencv.core.MatOfPoint2f;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import java.util.ArrayList;
import java.util.List;

public class AlignImages {

    public static List<Mat> alignImages(List<Mat> images) {
        List<Mat> alignedImages = new ArrayList<>();
        Mat rootImg = images.get(0).clone();
        alignedImages.add(rootImg);

        for (int i = 1; i < images.size(); i++) {
            Mat img = images.get(i);

            // Find correspondences between rootImg and img
            MatOfPoint2f pointsRoot = new MatOfPoint2f();
            MatOfPoint2f pointsImg = new MatOfPoint2f();
            findCorrespondences(rootImg, img, pointsRoot, pointsImg);

            // Compute transformation matrix (homography or affine)
            Mat transformMatrix = calculateTransform(pointsImg, pointsRoot, "rigid");

            // Warp image
            Mat warpedImg = warpImages(img, rootImg, transformMatrix);
            alignedImages.add(warpedImg);
        }

        return alignedImages;
    }

    private static void findCorrespondences(Mat img1, Mat img2, MatOfPoint2f points1, MatOfPoint2f points2) {
        // TODO: Implement feature matching to find correspondences (SIFT, ORB, etc.)
    }

    private static Mat calculateTransform(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints, String type) {
        Mat transformMatrix = new Mat();
        if ("rigid".equalsIgnoreCase(type)) {
            transformMatrix = Calib3d.estimateAffine2D(srcPoints, dstPoints);
        } else if ("affine".equalsIgnoreCase(type)) {
            transformMatrix = Imgproc.getAffineTransform(srcPoints, dstPoints);
        } else if ("homography".equalsIgnoreCase(type)) {
            transformMatrix = Calib3d.findHomography(srcPoints, dstPoints);
        }
        return transformMatrix;
    }

    private static Mat warpImages(Mat src, Mat reference, Mat transformMatrix) {
        Mat warpedImage = new Mat();
        Imgproc.warpAffine(src, warpedImage, transformMatrix, reference.size());
        return warpedImage;
    }
}


import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.Scalar;

public class ImageProcessor {

    public static Mat[] composite(Mat sharpImage, Mat blurredImage, List<Mat> flowMaps, Mat subjectMask) {
        // Calculate MFlow from flow maps and sharp image
        Mat MFlow = calcMFlow(flowMaps, sharpImage);
        MFlow = normalize(MFlow);

        // Combine flow mask with subject mask using max operation
        Mat flowFaceMask = new Mat(MFlow.size(), MFlow.type());
        Core.max(MFlow, subjectMask, flowFaceMask);
        flowFaceMask = normalize(flowFaceMask);

        // Perform alpha blending between sharp and blurred images
        Mat composite = alphaBlending(sharpImage, flowFaceMask, blurredImage);

        return new Mat[]{composite, flowFaceMask};
    }

    private static Mat calcMFlow(List<Mat> flowMaps, Mat sharpImage) {
        // TODO: Implement MFlow computation based on flow maps
        return new Mat(sharpImage.size(), sharpImage.type(), Scalar.all(0)); // Placeholder
    }

    private static Mat normalize(Mat image) {
        // TODO: Implement normalization function
        return image; // Placeholder
    }

    private static Mat alphaBlending(Mat sharpImage, Mat mask, Mat blurredImage) {
        Mat blendedImage = new Mat();
        Core.addWeighted(sharpImage, 1.0, blurredImage, 1.0, 0, blendedImage);
        return blendedImage;
    }
}

import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

public class ImageProcessor {

    public static Mat blurImages(List<Mat> images, List<Mat> flowMaps) {
        if (images == null || images.isEmpty()) {
            return new Mat();  // Return empty matrix if no images
        }

        Mat blurredImage = images.get(0).clone();  // Initial image
        double weight = 1.0;

        for (int i = 1; i < images.size(); i++) {
            // Generate in-between frames
            int NUM_FRAMES = (1 << 4) - 1;  // 15 intermediate frames
            List<Mat> inbetweenFrames = interpolateFrames(images.get(i - 1), images.get(i), flowMaps.get(i - 1), NUM_FRAMES);
            inbetweenFrames.add(images.get(i));  // Add next frame

            double newWeight = weight + inbetweenFrames.size();
            Mat sumFrames = Mat.zeros(blurredImage.size(), blurredImage.type());

            // Sum all intermediate frames
            for (Mat frame : inbetweenFrames) {
                Core.add(sumFrames, frame, sumFrames);
            }

            // Weighted blur: (weight * blurredImage + sumFrames) / newWeight
            Mat weightedBlurredImage = new Mat();
            Core.addWeighted(blurredImage, weight / newWeight, sumFrames, 1.0 / newWeight, 0, weightedBlurredImage);

            blurredImage = weightedBlurredImage;
            weight = newWeight;
        }

        return blurredImage;
    }
}

import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

public class ImageProcessor {

    public static List<Mat> interpolateFrames(Mat frame1, Mat frame2, Mat inputFlowMap, int numFrames) {
        List<Mat> inBetweenFrames = new ArrayList<>();

        for (int t = 1; t < numFrames; t++) {
            float scaleFactor = (float) t / numFrames;
            Mat interpolatedFrame = generateOneFrame(frame1, inputFlowMap, scaleFactor);
            inBetweenFrames.add(interpolatedFrame);
        }

        return inBetweenFrames;
    }

    private static Mat generateOneFrame(Mat frame, Mat flowMap, float t) {
        int h = frame.rows();
        int w = frame.cols();

        // Scale the optical flow
        Mat scaledFlow = new Mat();
        Core.multiply(flowMap, new Mat(flowMap.size(), flowMap.type(), Scalar.all(t)), scaledFlow);

        // Create mesh grid for pixel locations
        Mat mapX = new Mat(h, w, CvType.CV_32FC1);
        Mat mapY = new Mat(h, w, CvType.CV_32FC1);

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                double[] flow = scaledFlow.get(i, j);
                mapX.put(i, j, j + flow[0]);
                mapY.put(i, j, i + flow[1]);
            }
        }

        // Warp the frame using remapping
        Mat interpolatedFrame = new Mat();
        Imgproc.remap(frame, interpolatedFrame, mapX, mapY, Imgproc.INTER_LINEAR);

        return interpolatedFrame;
    }
}
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ImageProcessor {

    public static List<Mat> readImages(String directory, double resizeScale) {
        List<Mat> images = new ArrayList<>();

        File dir = new File(directory);
        if (!dir.exists() || !dir.isDirectory()) {
            return images;
        }

        List<String> filenames = new ArrayList<>(Arrays.asList(dir.list()));
        Collections.sort(filenames);

        for (String filename : filenames) {
            File file = new File(dir, filename);
            if (file.isDirectory() || filename.equals("aligned_images") || filename.equals("output") ||
                filename.equals("flow_map") || filename.equals("output_initial") || filename.equals("output_initial2")) {
                continue;
            }

            Mat img = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
            if (img.empty()) {
                continue;
            }

            if (resizeScale != 1) {
                Mat resizedImg = new Mat();
                Imgproc.resize(img, resizedImg, new Size(img.cols() * resizeScale, img.rows() * resizeScale));
                img = resizedImg;
            }

            images.add(img);
        }
        return images;
    }

    public static void writeImages(String directory, List<Mat> images) {
        File dir = new File(directory);
        if (!dir.exists()) {
            dir.mkdirs();  // Create directory if it does not exist
        }

        int numDigits = 3;
        for (int i = 0; i < images.size(); i++) {
            String filename = String.format("img_%0" + numDigits + "d.png", i);
            File outputFile = new File(dir, filename);
            Imgcodecs.imwrite(outputFile.getAbsolutePath(), images.get(i));
        }
    }

    public static List<Mat> getAlignedImages(List<Mat> images, boolean fromCache, String directory) {
        List<Mat> alignedImages = new ArrayList<>();

        // If cached images exist, read them
        if (fromCache && directory != null) {
            File dir = new File(directory);
            if (dir.exists() && dir.isDirectory() && dir.list().length > 0) {
                for (String filename : dir.list()) {
                    Mat img = Imgcodecs.imread(new File(dir, filename).getAbsolutePath());
                    if (!img.empty()) {
                        alignedImages.add(img);
                    }
                }
                return alignedImages;
            }
        }

        // Else, generate aligned images
        alignedImages = alignImages(images);  // Assuming alignImages() is implemented

        // Save aligned images if a directory is provided
        if (directory != null) {
            writeImages(directory, alignedImages);
        }

        return alignedImages;
    }

    // Placeholder for alignImages() function
    private static List<Mat> alignImages(List<Mat> images) {
        // Implement image alignment logic here
        return images;  // Return the same images for now
    }
}

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.List;

public class ImageProcessor {

    public static void pipeline() {
        // 0. Prepare directories
        String imageDirectory = "examples/tiger/";
        String flowmapDirectory = imageDirectory + "flow_map/";
        String alignedImagesDirectory = imageDirectory + "aligned_images/";
        String outputDirectory = imageDirectory + "output/";

        new File(flowmapDirectory).mkdirs();
        new File(alignedImagesDirectory).mkdirs();
        new File(outputDirectory).mkdirs();

        // 1.1 Read all images
        System.out.println("Reading Images...");
        List<Mat> images = readImages(imageDirectory, 1.0 / 8);
        System.out.println("Number of images: " + images.size() + " = N");
        System.out.println("Image shape: " + images.get(0).size() + " = (H, W, 3)");
        System.out.println();

        String method = "raft";

        // 1.2 Resize images (if method is raft)
        if (method.equals("raft")) {
            int H = images.get(0).rows();
            int W = images.get(0).cols();
            H -= H % 8;
            W -= W % 8;
            for (int i = 0; i < images.size(); i++) {
                Mat resizedImg = new Mat();
                Imgproc.resize(images.get(i), resizedImg, new Size(W, H));
                images.set(i, resizedImg);
            }
        }

        // 1.3 Align images using the first frame as reference
        System.out.println("Aligning Images...");
        images = getAlignedImages(images, false, alignedImagesDirectory);
        System.out.println("Number of aligned images: " + images.size() + " = N");
        System.out.println("Aligned image shape FIRST: " + images.get(0).size() + " = (H, W, 3)");
        System.out.println("Aligned image shape LAST: " + images.get(images.size() - 1).size() + " = (H, W, 3)");

        // 1.4 Naive long exposure
        System.out.println("Naively blurring images...");
        Mat naiveBlurred = naiveBlurImages(images);
        Imgcodecs.imwrite(outputDirectory + "naive_blurred.png", naiveBlurred);
        System.out.println();

        // 3. Subject detection
        System.out.println("Creating face mask...");
        Mat sharpImage = images.get(0);
        Mat subjectMask = null;

        boolean USE_USER_MASK = false; // Change as needed

        if (USE_USER_MASK) {
            subjectMask = getMask(sharpImage);
        } else {
            subjectMask = subjectDetection(sharpImage);
        }

        Imgcodecs.imwrite(outputDirectory + "face_mask.png", subjectMask);
        System.out.println();

        // 2. Read/calculate optical flow maps
        System.out.println("Calculating optical flow maps...");
        List<Mat> flowMaps = calculateOpticalFlow(images, method, false, flowmapDirectory);
        Imgcodecs.imwrite(outputDirectory + "example_flow_map.png", flowMaps.get(0));
        System.out.println("Number of flow maps: " + flowMaps.size() + " = N-1");
        System.out.println("Flow shape: " + flowMaps.get(0).size() + " = (H, W, 2)");
        System.out.println();

        // 4. Interpolate between frames -> one blurred image
        System.out.println("Linearly interpolating between frames...");
        Mat blurredImage = blurImages(images, flowMaps);
        Imgcodecs.imwrite(outputDirectory + "blurred_image.png", blurredImage);
        System.out.println();

        // 5. Composite
        System.out.println("Compositing...");
        Mat[] compositeResults = composite(sharpImage, blurredImage, flowMaps, subjectMask);
        Imgcodecs.imwrite(outputDirectory + "result.png", compositeResults[0]);
        Imgcodecs.imwrite(outputDirectory + "flow_face_mask.png", compositeResults[1]);
        System.out.println("Finished!");
    }

    public static void main(String[] args) {
        pipeline();
    }

    // Placeholder functions to be implemented

    public static Mat naiveBlurImages(List<Mat> images) {
        // Implement naive blurring logic here
        return images.get(0); // Placeholder
    }

    public static Mat getMask(Mat image) {
        // Implement mask detection logic
        return image; // Placeholder
    }

    public static Mat subjectDetection(Mat image) {
        // Implement subject detection logic (e.g., face detection)
        return image; // Placeholder
    }

    public static List<Mat> calculateOpticalFlow(List<Mat> images, String method, boolean fromCache, String flowmapDir) {
        // Implement optical flow calculation logic
        return images; // Placeholder
    }

    public static Mat blurImages(List<Mat> images, List<Mat> flowMaps) {
        // Implement interpolation logic for blurring
        return images.get(0); // Placeholder
    }

    public static Mat[] composite(Mat sharpImage, Mat blurredImage, List<Mat> flowMaps, Mat subjectMask) {
        // Implement compositing logic
        return new Mat[]{sharpImage, subjectMask}; // Placeholder
    }
}

import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.video.Video;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ImageProcessor {

    public static List<Mat> calculateOpticalFlow(List<Mat> images, String method, boolean fromCache, String flowmapDir) {
        List<Mat> flowMaps = new ArrayList<>();

        if (flowmapDir != null) {
            flowmapDir = flowmapDir + "/" + method;
            new File(flowmapDir).mkdirs(); // Create directory if not exists
        }

        // Load from cache if possible
        File flowDir = new File(flowmapDir);
        if (fromCache && flowDir.exists() && flowDir.isDirectory() && flowDir.list().length > 0) {
            for (String filename : flowDir.list()) {
                Mat flowmap = Imgcodecs.imread(new File(flowDir, filename).getAbsolutePath(), Imgcodecs.IMREAD_UNCHANGED);
                if (!flowmap.empty()) {
                    flowMaps.add(flowmap);
                }
            }
            return flowMaps;
        }

        // If cache not available, compute optical flow
        for (int i = 0; i < images.size() - 1; i++) {
            Mat prevGray = new Mat();
            Mat nextGray = new Mat();
            Mat flow = new Mat(images.get(i).size(), CvType.CV_32FC2); // Optical flow map

            // Convert images to grayscale
            Imgproc.cvtColor(images.get(i), prevGray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.cvtColor(images.get(i + 1), nextGray, Imgproc.COLOR_BGR2GRAY);

            // Compute optical flow using Farneback method
            Video.calcOpticalFlowFarneback(prevGray, nextGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

            // Save flow map
            String flowFilename = String.format(flowmapDir + "/flow_%02d.png", i);
            Imgcodecs.imwrite(flowFilename, flow);

            flowMaps.add(flow);
        }

        return flowMaps;
    }
}

import org.opencv.core.Mat;
import org.opencv.core.Core;

public class ImageProcessor {

    public static Mat subjectDetection(Mat image, boolean faceEnable) {
        System.out.println("Finding gaze attention mask...");
        Mat attentionMask = getAttentionMask(image);
        attentionMask = normalize(attentionMask);

        if (!faceEnable) {
            return attentionMask;
        }

        System.out.println("Finding head mask...");
        Mat headMask = getHeadSegmentation(image);
        headMask = normalize(headMask);

        // Combine both masks: face_mask = attention_mask * (1 + head_mask)
        Mat faceMask = new Mat();
        Core.add(headMask, new Mat(headMask.size(), headMask.type(), Scalar.all(1)), headMask); // (1 + head_mask)
        Core.multiply(attentionMask, headMask, faceMask); // attention_mask * (1 + head_mask)
        
        faceMask = normalize(faceMask);
        return faceMask;
    }

    private static Mat getAttentionMask(Mat image) {
        // TODO: Implement attention mask calculation
        return new Mat(image.size(), image.type()); // Placeholder
    }

    private static Mat getHeadSegmentation(Mat image) {
        // TODO: Implement head segmentation logic
        return new Mat(image.size(), image.type()); // Placeholder
    }

    private static Mat normalize(Mat image) {
        // TODO: Implement normalization function
        return image; // Placeholder
    }
}
