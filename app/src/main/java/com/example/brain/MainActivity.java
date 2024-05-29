package com.example.brain;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.brain.R;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_SELECT_IMAGE = 1;
    private static final int REQUEST_CODE_PERMISSION = 2;

    private Interpreter tflite;
    private ImageView imageView;
    private TextView textViewResult;
    private Bitmap selectedBitmap;

    private static final String[] labels = {
            "glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textViewResult = findViewById(R.id.textViewResult);
        Button buttonSelectImage = findViewById(R.id.buttonSelectImage);
        Button buttonClassify = findViewById(R.id.buttonClassify);

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Model could not be loaded", Toast.LENGTH_SHORT).show();
        }

        buttonSelectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Request the permission to read external storage
                if (ContextCompat.checkSelfPermission(MainActivity.this, android.Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_CODE_PERMISSION);
                } else {
                    selectImage();
                }
            }
        });

        buttonClassify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (selectedBitmap != null) {
                    classifyImage();
                } else {
                    Toast.makeText(MainActivity.this, "Please select an image first", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void selectImage() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_CODE_SELECT_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE_SELECT_IMAGE && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            try {
                InputStream imageStream = getContentResolver().openInputStream(imageUri);
                selectedBitmap = BitmapFactory.decodeStream(imageStream);
                imageView.setImageBitmap(selectedBitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        try (FileInputStream fileInputStream = new FileInputStream(getAssets().openFd("finalyear.tflite").getFileDescriptor());
             FileChannel fileChannel = fileInputStream.getChannel()) {
            long startOffset = getAssets().openFd("finalyear.tflite").getStartOffset();
            long declaredLength = getAssets().openFd("finalyear.tflite").getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    private void classifyImage() {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(selectedBitmap, 150, 150, true);
        ByteBuffer inputBuffer = convertBitmapToByteBuffer(resizedBitmap);

        float[][] result = new float[1][4];
        if (tflite != null) {
            tflite.run(inputBuffer, result);

            int predictedLabelIndex = getMaxIndex(result[0]);
            String predictedLabel = labels[predictedLabelIndex];
            float confidence = result[0][predictedLabelIndex] * 100;

            textViewResult.setText("Prediction: " + predictedLabel + "\nConfidence: " + confidence + "%");
        } else {
            Toast.makeText(this, "Model is not initialized", Toast.LENGTH_SHORT).show();
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 150 * 150 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[150 * 150];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < 150; ++i) {
            for (int j = 0; j < 150; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.f);
                byteBuffer.putFloat((val & 0xFF) / 255.f);
            }
        }

        return byteBuffer;
    }

    private int getMaxIndex(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
