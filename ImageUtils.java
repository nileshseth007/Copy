import org.opencv.core.CvType;
import org.opencv.core.Mat;
import java.io.*;

public class ImageUtils {

    // Load a .npy file into an OpenCV Mat
    public static Mat loadNpyAsMat(String filePath) {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filePath))) {
            int rows = dis.readInt();
            int cols = dis.readInt();
            int type = dis.readInt();

            Mat mat = new Mat(rows, cols, type);
            byte[] data = new byte[rows * cols * (int) mat.elemSize()];
            dis.readFully(data);
            mat.put(0, 0, data);
            return mat;
        } catch (IOException e) {
            throw new RuntimeException("Error loading .npy file: " + filePath, e);
        }
    }

    // Save an OpenCV Mat as a .npy file
    public static void saveMatAsNpy(String filePath, Mat mat) {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filePath))) {
            dos.writeInt(mat.rows());
            dos.writeInt(mat.cols());
            dos.writeInt(mat.type());
            byte[] data = new byte[(int) (mat.total() * mat.elemSize())];
            mat.get(0, 0, data);
            dos.write(data);
        } catch (IOException e) {
            throw new RuntimeException("Error saving .npy file: " + filePath, e);
        }
    }
}
