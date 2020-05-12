package com.example.javacvtest;

import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Rational;
import android.view.Surface;
import android.view.TextureView;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.util.ArrayList;

import static org.opencv.core.CvType.CV_32SC1;

class LimitedSizeQueue<K> extends ArrayList<K> {

    private int maxSize;

    public LimitedSizeQueue(int size) {
        this.maxSize = size;
    }

    public boolean add(K k) {
        boolean r = super.add(k);
        if (size() > maxSize) {
            removeRange(0, size() - maxSize);
        }
        return r;
    }

    public K getYoungest() {
        return get(size() - 1);
    }

    public K getOldest() {
        return get(0);
    }
}

public class MainActivity extends AppCompatActivity {

    private int REQUEST_CODE_PERMISSIONS = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    TextureView textureView;
    ImageView ivBitmap;
    private CascadeClassifier cascadeClassifier;
    private opencv_face.FaceRecognizer mLBPHFaceRecognizer;
    private Mat grayScaleMat;
    ImageAnalysis imageAnalysis;
    Preview preview;
    TextView status;
    int maxFaceCount = 10;
    ArrayList<opencv_core.Mat> capturedFaces = new ArrayList<>(maxFaceCount);
    public static final String TAG = "LogCat";
    LimitedSizeQueue<Integer> prevConfs;

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    initializeOpenCVDependencies();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    private void initializeOpenCVDependencies() {

        try {
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "cascade.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            grayScaleMat = new Mat();
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            prevConfs = new LimitedSizeQueue<Integer>(20);

            if (allPermissionsGranted()) {
                startCamera();
            } else {
                ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
            }

        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textureView = findViewById(R.id.textureView);
        ivBitmap = findViewById(R.id.ivBitmap);
        status = findViewById(R.id.status);

    }

    private void startCamera() {

        CameraX.unbindAll();
        preview = setPreview();
        imageAnalysis = setImageAnalysis();

        CameraX.bindToLifecycle(this, preview, imageAnalysis);
    }


    private Preview setPreview() {

        Rational aspectRatio = new Rational(textureView.getWidth(), textureView.getHeight());
        android.util.Size screen = new android.util.Size(textureView.getWidth(), textureView.getHeight());

        PreviewConfig pConfig = new PreviewConfig.Builder().setTargetAspectRatio(aspectRatio).setTargetResolution(screen).build();
        Preview preview = new Preview(pConfig);

        preview.setOnPreviewOutputUpdateListener(
                new Preview.OnPreviewOutputUpdateListener() {
                    @Override
                    public void onUpdated(Preview.PreviewOutput output) {
                        ViewGroup parent = (ViewGroup) textureView.getParent();
                        parent.removeView(textureView);
                        parent.addView(textureView, 0);

                        textureView.setSurfaceTexture(output.getSurfaceTexture());
                        updateTransform();
                    }
                });

        return preview;
    }

    private ImageAnalysis setImageAnalysis() {

        HandlerThread analyzerThread = new HandlerThread("OpenCVAnalysis");
        analyzerThread.start();


        ImageAnalysisConfig imageAnalysisConfig = new ImageAnalysisConfig.Builder()
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .setCallbackHandler(new Handler(analyzerThread.getLooper()))
                .setImageQueueDepth(1).build();

        ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);

        imageAnalysis.setAnalyzer(
                new ImageAnalysis.Analyzer() {
                    @Override
                    public void analyze(ImageProxy image, int rotationDegrees) {
                        final Bitmap bitmap = textureView.getBitmap();
                        if (bitmap == null)
                            return;

                        Mat mat = new Mat();
                        Utils.bitmapToMat(bitmap, mat);
                        Imgproc.cvtColor(mat, grayScaleMat, Imgproc.COLOR_RGBA2GRAY);
                        MatOfRect faces = new MatOfRect();

                        cascadeClassifier.detectMultiScale(grayScaleMat, faces, 1.1, 5, 2,
                                new Size(100, 100), new Size());

                        Rect[] faceRects = faces.toArray();
                        if (faceRects.length > 0) {
                            Rect faceRect = faceRects[0];
                            Imgproc.rectangle(mat, faceRect.tl(), faceRect.br(), new Scalar(255, 0, 0, 255), 3);

                            final Mat faceMat = new Mat(grayScaleMat, faceRect);
                            //Imgproc.equalizeHist(faceMat, faceMat);
                            Imgproc.resize(faceMat, faceMat, new Size(100, 100));

                            opencv_core.Mat javaCvFaceMat = new opencv_core.Mat((Pointer) null) {{
                                address = faceMat.getNativeObjAddr();
                            }};

                            if (mLBPHFaceRecognizer == null) {
                                capturedFaces.add(javaCvFaceMat);
                                runOnUiThread(() -> {
                                    status.setText("face frames left to train: " + (maxFaceCount - capturedFaces.size()));
                                });
                            } else {
                                int[] labels = new int[1];
                                double[] confidences = new double[1];
                                mLBPHFaceRecognizer.predict(javaCvFaceMat, labels, confidences);

                                int predictedLabel = labels[0];
                                int acceptanceLevel = (int) confidences[0];
                                prevConfs.add(acceptanceLevel);


                                int posX = (int) Math.max(faceRect.tl().x - 10, 0);
                                int posY = (int) Math.max(faceRect.tl().y - 10, 0);
//                                StringBuilder nums = new StringBuilder();
//                                for (int num : prevConfs) {
//                                    nums.append(num).append(",");
//                                }
//                                Log.d(TAG, nums + " " + getAvgFromPrevConfs(prevConfs));
                                Imgproc.putText(mat, acceptanceLevel + "/" + getAvgFromPrevConfs(prevConfs), new Point(posX, posY),
                                        Core.FONT_HERSHEY_TRIPLEX, 1.5, new Scalar(255, 255, 0, 255), 3);
                            }


                        }

                        if (mLBPHFaceRecognizer == null && capturedFaces.size() >= maxFaceCount) {
                            runOnUiThread(() -> {
                                status.setText("training...");
                            });
                            opencv_core.Mat labels = new opencv_core.Mat(capturedFaces.size(), 1, CV_32SC1);
                            opencv_core.MatVector faceImages = new opencv_core.MatVector(capturedFaces.size());
                            mLBPHFaceRecognizer = opencv_face.LBPHFaceRecognizer.create();
                            IntBuffer intBuffer = labels.createBuffer();

                            for (int i = 0; i < capturedFaces.size(); i++) {
                                faceImages.put(i, capturedFaces.get(i));
                                intBuffer.put(i, 1);
                            }
                            mLBPHFaceRecognizer.train(faceImages, labels);
                            runOnUiThread(() -> {
                                status.setText("recognition...");
                            });
                        }

                        Utils.matToBitmap(mat, bitmap);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                ivBitmap.setImageBitmap(bitmap);
                            }
                        });
                    }
                });


        return imageAnalysis;

    }

    private int getAvgFromPrevConfs(LimitedSizeQueue<Integer> queue) {
        float totalSum = 0;
        for (float conf : queue) {
            totalSum += conf;
        }

        return (int) (totalSum / queue.size());
    }

    private void updateTransform() {
        Matrix mx = new Matrix();
        float w = textureView.getMeasuredWidth();
        float h = textureView.getMeasuredHeight();

        float cX = w / 2f;
        float cY = h / 2f;

        int rotationDgr;
        int rotation = (int) textureView.getRotation();

        switch (rotation) {
            case Surface.ROTATION_0:
                rotationDgr = 0;
                break;
            case Surface.ROTATION_90:
                rotationDgr = 90;
                break;
            case Surface.ROTATION_180:
                rotationDgr = 180;
                break;
            case Surface.ROTATION_270:
                rotationDgr = 270;
                break;
            default:
                return;
        }

        mx.postRotate((float) rotationDgr, cX, cY);
        textureView.setTransform(mx);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {

        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }


}
