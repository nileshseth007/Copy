import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.DISOpticalFlow;
import java.util.ArrayList;
import java.util.List;

public class Raft {

    public static List<Mat> calculateRaftOpticalFlow(List<Mat> images) {
        List<Mat> predictedFlows = new ArrayList<>();

        // Create DIS Optical Flow instance (equivalent to RAFT in OpenCV)
        DISOpticalFlow disFlow = DISOpticalFlow.create(DISOpticalFlow.PRESET_ULTRAFAST);

        for (int i = 0; i < images.size() - 1; i++) {
            Mat flow = new Mat();
            disFlow.calc(images.get(i), images.get(i + 1), flow);
            predictedFlows.add(flow);
        }

        return predictedFlows;
    }
}
