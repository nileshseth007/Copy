import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV initialization failed.", Toast.LENGTH_LONG).show();
            return;
        }

        Button runPipelineButton = findViewById(R.id.runPipelineButton);
        runPipelineButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runPipeline();
            }
        });
    }

    private void runPipeline() {
        new Thread(() -> {
            try {
                String imageDirectory = "examples"; // No "assets/" prefix needed
                String flowmapDirectory = imageDirectory + "/flow_map/";
                String alignedImagesDirectory = imageDirectory + "/aligned_images/";
                String outputDirectory = imageDirectory + "/output/";

                // Load images from assets
                List<Mat> images = ImageUtils.readImages(getApplicationContext(), imageDirectory, 1.0 / 8);

                if (images.isEmpty()) {
                    runOnUiThread(() -> Toast.makeText(getApplicationContext(), "No images found!", Toast.LENGTH_LONG).show());
                    return;
                }

                // Run the pipeline function
                ImageProcessor.pipeline(images, flowmapDirectory, alignedImagesDirectory, outputDirectory);

                runOnUiThread(() -> Toast.makeText(getApplicationContext(), "Pipeline execution completed!", Toast.LENGTH_LONG).show());
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> Toast.makeText(getApplicationContext(), "Error in pipeline execution!", Toast.LENGTH_LONG).show());
            }
        }).start();
    }
}
