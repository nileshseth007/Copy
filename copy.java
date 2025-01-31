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
