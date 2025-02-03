import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class SubjectDetection {

    public static Mat getHeadSegmentation(Mat image) {
        // Load OpenCV Haar Cascade face detector
        CascadeClassifier faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");

        // Convert image to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

        // Detect faces
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, new Size(30, 30), new Size());

        // Create a black mask
        Mat headMask = Mat.zeros(gray.size(), CvType.CV_8UC1);

        // Draw white rectangles on the mask for detected faces
        for (Rect rect : faces.toArray()) {
            Imgproc.rectangle(headMask, rect.tl(), rect.br(), new Scalar(255), -1);
        }

        return headMask;
    }
  public static Mat getAttentionMask(Mat image) {
    // Load OpenCV Haar Cascade eye detector
    CascadeClassifier eyeCascade = new CascadeClassifier("haarcascade_eye.xml");

    // Convert image to grayscale
    Mat gray = new Mat();
    Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

    // Detect eyes
    MatOfRect eyes = new MatOfRect();
    eyeCascade.detectMultiScale(gray, eyes, 1.1, 10, 0, new Size(20, 20), new Size());

    // Create an empty black mask
    Mat attentionMask = Mat.zeros(gray.size(), CvType.CV_8UC1);

    // Draw white circular masks over detected eyes
    for (Rect rect : eyes.toArray()) {
        Point center = new Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
        int radius = Math.max(rect.width, rect.height) / 2;
        Imgproc.circle(attentionMask, center, radius, new Scalar(255), -1);
    }

    return attentionMask;
}
  public static Mat normalize(Mat mask) {
    Mat normalizedMask = new Mat();
    Core.normalize(mask, normalizedMask, 0, 1, Core.NORM_MINMAX);
    return normalizedMask;
}

  
}
