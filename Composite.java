import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import java.io.File;
import java.util.Arrays;

// Mat flowFaceMask = Composite.calcMFlow(flowMaps, sharpImage);


public class Composite {

  public static Mat calcFRef(Mat F) {
      Core.MinMaxLocResult minMax = Core.minMaxLoc(F);
      double robustMax = minMax.maxVal;
  
      Mat FRef = Mat.zeros(F.size(), F.type());
      FRef.setTo(new Scalar(robustMax));
  
      return FRef;
  }

  public static Mat calcF(Mat[] opticalFlows) {
      List<Mat> magList = new ArrayList<>();
  
      for (Mat opticalFlow : opticalFlows) {
          Mat norm = new Mat();
          Core.norm(opticalFlow, norm, Core.NORM_L2);
          magList.add(norm);
      }
  
      Mat opticalFlowsArray = new Mat();
      Core.max(magList.toArray(new Mat[0]), opticalFlowsArray);
      
      return opticalFlowsArray;
  }

  public static Mat calcMFlow(Mat[] opticalFlows, Mat sharpImage) {
      Mat F = calcF(opticalFlows);
      Mat FRef = calcFRef(F);
  
      Mat alphaFRef = new Mat();
      Core.multiply(FRef, new Scalar(ALPHA), alphaFRef);
      Core.max(alphaFRef, new Scalar(0), alphaFRef);
  
      Mat betaFRef = new Mat();
      Core.multiply(FRef, new Scalar(BETA), betaFRef);
      Core.min(betaFRef, new Scalar(1), betaFRef);
  
      Mat numerator = new Mat();
      Core.subtract(F, alphaFRef, numerator);
  
      Mat denominator = new Mat();
      Core.subtract(betaFRef, alphaFRef, denominator);
  
      Mat mFlow = new Mat();
      Core.divide(numerator, denominator, mFlow);
  
      Mat bilateral = new Mat();
      Imgproc.bilateralFilter(mFlow, bilateral, 15, 75, 75);
  
      return bilateral;
  }


  public static Mat alphaBlending(Mat source, Mat mask, Mat target) {
      Mat mask3Channel = new Mat();
      List<Mat> maskChannels = new ArrayList<>();
      maskChannels.add(mask);
      maskChannels.add(mask);
      maskChannels.add(mask);
      Core.merge(maskChannels, mask3Channel);
  
      Mat maskedSource = new Mat();
      Core.multiply(source, mask3Channel, maskedSource);
  
      Mat inverseMask = new Mat();
      Core.subtract(Mat.ones(mask.size(), mask.type()), mask, inverseMask);
  
      Mat maskedTarget = new Mat();
      Core.multiply(target, inverseMask, maskedTarget);
  
      Mat blended = new Mat();
      Core.add(maskedSource, maskedTarget, blended);
  
      return blended;
  }
public static Mat poissonBlend(Mat source, Mat mask, Mat target) {
    int rows = source.rows();
    int cols = source.cols();
    Mat blended = new Mat(source.size(), source.type());

    Mat channel0 = new Mat(), channel1 = new Mat(), channel2 = new Mat();
    List<Mat> sourceChannels = new ArrayList<>();
    List<Mat> targetChannels = new ArrayList<>();
    Core.split(source, sourceChannels);
    Core.split(target, targetChannels);

    Mat A = new Mat(); // Sparse matrix for solving
    channel0 = poissonBlendChannel(sourceChannels.get(0), mask, targetChannels.get(0), null, true, false);
    channel1 = poissonBlendChannel(sourceChannels.get(1), mask, targetChannels.get(1), A, false, false);
    channel2 = poissonBlendChannel(sourceChannels.get(2), mask, targetChannels.get(2), A, false, false);

    List<Mat> blendedChannels = Arrays.asList(channel0, channel1, channel2);
    Core.merge(blendedChannels, blended);
    
    return blended;
}
  public static Mat poissonBlendChannel(Mat source, Mat mask, Mat target, Mat A, boolean makeA, boolean isAlpha) {
      int numRows = source.rows();
      int numCols = source.cols();
      Mat x = Mat.zeros(source.size(), source.type());
  
      if (makeA) {
          A = Mat.eye(numRows * numCols, numRows * numCols, CvType.CV_32F); // Identity matrix
      }
  
      Mat b = Mat.zeros(numRows * numCols, 1, CvType.CV_32F);
  
      for (int col = 0; col < numCols; col++) {
          for (int row = 0; row < numRows; row++) {
              int oneDimIndex = row * numCols + col;
              if (mask.get(row, col)[0] == 0) {
                  b.put(oneDimIndex, 0, target.get(row, col)[0]);
              } else {
                  if (makeA) {
                      boundsCheckA(A, oneDimIndex, numRows, numCols, col, row, source);
                  }
                  double sourceGrad = boundsCheckB(b, oneDimIndex, numRows, numCols, col, row, source);
                  if (isAlpha) {
                      double targetGrad = boundsCheckB(b, oneDimIndex, numRows, numCols, col, row, target);
                      b.put(oneDimIndex, 0, Math.max(sourceGrad, targetGrad));
                  } else {
                      b.put(oneDimIndex, 0, sourceGrad);
                  }
              }
          }
      }
  
      Core.solve(A, b, x);
      return x.reshape(1, numRows);
  }

  public static List<Mat> buildLaplacianPyramidRecursive(Mat img, int depth, List<Mat> detailList) {
      int MAX_DEPTH = 4;
      if (depth >= MAX_DEPTH) {
          detailList.add(img);
          return detailList;
      }
      Mat blur = new Mat();
      Imgproc.GaussianBlur(img, blur, new Size(3, 3), 0);
      
      Mat detail = new Mat();
      Core.subtract(img, blur, detail);
      detailList.add(detail);
  
      Mat downsampled = new Mat();
      Imgproc.resize(blur, downsampled, new Size(), 0.5, 0.5, Imgproc.INTER_LINEAR);
  
      return buildLaplacianPyramidRecursive(downsampled, depth + 1, detailList);
  }

  public static List<Mat> buildGaussianPyramidRecursive(Mat img, int depth, List<Mat> blurList) {
      int MAX_DEPTH = 4;
      if (depth >= MAX_DEPTH) {
          blurList.add(img);
          return blurList;
      }
      Mat blur = new Mat();
      Imgproc.GaussianBlur(img, blur, new Size(3, 3), 0);
      
      Mat downsampled = new Mat();
      Imgproc.resize(blur, downsampled, new Size(), 0.5, 0.5, Imgproc.INTER_LINEAR);
  
      blurList.add(blur);
      return buildGaussianPyramidRecursive(downsampled, depth + 1, blurList);
  }

  public static Mat reconstructLaplacianPyramid(List<Mat> Ls) {
      Mat image = Ls.get(Ls.size() - 1);
      for (int i = Ls.size() - 2; i >= 0; i--) {
          Mat upsampled = new Mat();
          Imgproc.resize(image, upsampled, new Size(Ls.get(i).width(), Ls.get(i).height()), 0, 0, Imgproc.INTER_LINEAR);
          Core.add(upsampled, Ls.get(i), image);
      }
      return image;
  }

  public static Mat laplacianPyramidBlend(Mat source, Mat mask, Mat target) {
      List<Mat> l1 = buildLaplacianPyramidRecursive(source, 1, new ArrayList<>());
      List<Mat> l2 = buildLaplacianPyramidRecursive(target, 1, new ArrayList<>());
      List<Mat> gm = buildGaussianPyramidRecursive(mask, 1, new ArrayList<>());
  
      List<Mat> lOut = new ArrayList<>();
      for (int i = 0; i < 4; i++) {
          Mat gm3Channel = new Mat();
          Core.merge(Arrays.asList(gm.get(i), gm.get(i), gm.get(i)), gm3Channel);
  
          Mat blended = new Mat();
          Core.addWeighted(l1.get(i), 1.0, l2.get(i), 1.0, 0.0, blended);
          Core.multiply(gm3Channel, l1.get(i), blended);
          Core.multiply(gm3Channel.inv(), l2.get(i), blended, 1.0, -1);
          lOut.add(blended);
      }
  
      return reconstructLaplacianPyramid(lOut);
  }

  public static double boundsCheckB(Mat b, int oneDimIndex, int numRows, int numCols, int col, int row, Mat source) {
      double sourceGrad = 4 * source.get(row, col)[0];
  
      if (row + 1 < numRows) {
          sourceGrad -= source.get(row + 1, col)[0];
      }
  
      if (row - 1 >= 0) {
          sourceGrad -= source.get(row - 1, col)[0];
      }
  
      if (col + 1 < numCols) {
          sourceGrad -= source.get(row, col + 1)[0];
      }
  
      if (col - 1 >= 0) {
          sourceGrad -= source.get(row, col - 1)[0];
      }
  
      return sourceGrad;
  }


  public static void boundsCheckA(Mat A, int oneDimIndex, int numRows, int numCols, int col, int row, Mat source) {
      int width = source.cols();
  
      if (row + 1 < numRows) {
          A.put(oneDimIndex, (row + 1) * width + col, -1);
      }
  
      if (row - 1 >= 0) {
          A.put(oneDimIndex, (row - 1) * width + col, -1);
      }
  
      if (col + 1 < numCols) {
          A.put(oneDimIndex, row * width + col + 1, -1);
      }
  
      if (col - 1 >= 0) {
          A.put(oneDimIndex, row * width + col - 1, -1);
      }
  
      A.put(oneDimIndex, oneDimIndex, 4);
  }

public static void composite(String sourceDir, int sharpImageIdx, String maskDir, String targetDir) {
      File sourceFolder = new File(sourceDir);
      File[] sourceFiles = sourceFolder.listFiles();
      Arrays.sort(sourceFiles); // Ensure files are sorted like in Python

      Mat sharpImage = null;
      int currIdx = 0;

      for (File file : sourceFiles) {
          if (currIdx == sharpImageIdx) {
              sharpImage = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
              break;
          }
          currIdx++;
      }

      if (sharpImage == null) {
          throw new RuntimeException("Sharp image could not be loaded!");
      }

      Mat mask = Imgcodecs.imread(maskDir + "/flow_face_mask.png", Imgcodecs.IMREAD_GRAYSCALE);
      Mat blur = Imgcodecs.imread(targetDir + "/blurred_image.png", Imgcodecs.IMREAD_COLOR);

      // Convert BGR to RGB
      Imgproc.cvtColor(sharpImage, sharpImage, Imgproc.COLOR_BGR2RGB);
      Imgproc.cvtColor(blur, blur, Imgproc.COLOR_BGR2RGB);

      // Convert uint8 to float32
      Mat source = new Mat();
      Mat target = new Mat();
      Mat maskFloat = new Mat();

      sharpImage.convertTo(source, CvType.CV_32F, 1.0 / 255.0);
      blur.convertTo(target, CvType.CV_32F, 1.0 / 255.0);
      mask.convertTo(maskFloat, CvType.CV_32F, 1.0 / 255.0);
      Core.round(maskFloat, maskFloat); // Equivalent to np.round(mask)

      // Create 3-channel mask
      List<Mat> maskChannels = Arrays.asList(maskFloat, maskFloat, maskFloat);
      Mat maskForComposite = new Mat();
      Core.merge(maskChannels, maskForComposite);

      // Perform alpha blending
      Mat alphaResult = alphaBlending(source, maskForComposite, target);
      Imgcodecs.imwrite(targetDir + "/result_alpha_blend.png", alphaResult);

      // Perform Poisson blending
      Mat poissonResult = poissonBlend(source, maskForComposite, target);
      Imgcodecs.imwrite(targetDir + "/result_poisson_blend.png", poissonResult);
  }



}
