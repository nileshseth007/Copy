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
