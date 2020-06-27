package ozgur.com.plantservice;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.karumi.dexter.Dexter;
import com.karumi.dexter.PermissionToken;
import com.karumi.dexter.listener.PermissionDeniedResponse;
import com.karumi.dexter.listener.PermissionGrantedResponse;
import com.karumi.dexter.listener.PermissionRequest;
import com.karumi.dexter.listener.single.PermissionListener;
import com.squareup.picasso.Picasso;

import java.io.File;
import java.io.FileInputStream;
import java.net.URISyntaxException;

import okhttp3.MultipartBody;
import ozgur.com.plantservice.Retrofit.IUploadAPI;
import ozgur.com.plantservice.Retrofit.RetrofitClient;
import ozgur.com.plantservice.Utils.Common;
import ozgur.com.plantservice.Utils.IUploadCallbacks;
import ozgur.com.plantservice.Utils.ProgressRequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity implements IUploadCallbacks {

    private static final int PICK_FILE_REQUEST = 1000;
    IUploadAPI mService;
    Button btnUpload;
    ImageView imageView;
    Uri selectedFileUri;
    TextView textView,textView2;

    ProgressDialog dialog;

    private IUploadAPI getUploadApi(){
        return RetrofitClient.getClient().create(IUploadAPI.class);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Dexter.withActivity(this).withPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
                .withListener(new PermissionListener() {
                    @Override
                    public void onPermissionGranted(PermissionGrantedResponse response) {
                        Toast.makeText(MainActivity.this,"Permission Accept",Toast.LENGTH_SHORT).show();
                    }

                    @Override
                    public void onPermissionDenied(PermissionDeniedResponse response) {
                        Toast.makeText(MainActivity.this,"You should Accept permission",Toast.LENGTH_SHORT).show();
                    }

                    @Override
                    public void onPermissionRationaleShouldBeShown(PermissionRequest permission, PermissionToken token) {

                    }
                }).check();

        mService = getUploadApi();

        btnUpload = (Button)findViewById(R.id.btn_upload);

        imageView = (ImageView)findViewById(R.id.image_view);

        textView = (TextView)findViewById(R.id.textView);

        textView2 = (TextView)findViewById(R.id.textView2);

        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                chooseFile();
            }
        });

        btnUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                uploadFile();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == Activity.RESULT_OK){
            if (requestCode == PICK_FILE_REQUEST){
                if (data != null){
                    selectedFileUri = data.getData();
                    if (selectedFileUri != null && !selectedFileUri.getPath().isEmpty())
                        imageView.setImageURI(selectedFileUri);
                    else
                        Toast.makeText(this,"Foto bulunamadi", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    private void uploadFile() {
        if (selectedFileUri != null) {
            dialog = new ProgressDialog(MainActivity.this);
            dialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
            dialog.setMessage("Uploading....");
            dialog.setIndeterminate(false);
            dialog.setMax(100);
            dialog.setCancelable(false);
            dialog.show();

            File file = null;
            try {
                file = new File(Common.getFilePath(this, selectedFileUri));
            } catch (URISyntaxException e) {
                e.printStackTrace();
            }

            if (file != null) {
                final ProgressRequestBody requestBody = new ProgressRequestBody(file, this);

                final MultipartBody.Part body = MultipartBody.Part.createFormData("image", file.getName(), requestBody);

                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        mService.uploadFile(body).enqueue(new Callback<String>() {
                            @Override
                            public void onResponse(Call<String> call, Response<String> response) {
                                dialog.dismiss();
                                String image_processed_link = new StringBuilder(response.body()).toString();
                                String[] res = image_processed_link.split("\\\\");
                                String[] res2 = res[2].split(" ");
                                if(res2[0].equalsIgnoreCase("dandelion")) {
                                    textView.setText(res2[0].toUpperCase());
                                    textView2.setText("Dandelion(Karahindiba) olarak da bilinen Taraxacum officinale, zaman zaman salata yeşilliği olarakda kullanılan sebzedir." +
                                            "Karahindiba yutulduğunda diüretik(vücuttan su atıcı) bir etkiye sahiptir. Potasyum için iyi bir kaynaktır.");
                                }else if(res2[0].equalsIgnoreCase("daisy")){
                                    textView.setText(res2[0].toUpperCase());
                                    textView2.setText("African Daisy(Afrika Papatyası Çiçeği): Güneşi seven afrika papatyası hemen her türlü toprak tipinde yetişebilen, park ve bahçeler için uygun," +
                                            " hastalıklara karşı ve susuzluğa karşı mukavemeti yüksek, bulunduğu ortamda kelebekleri hemen kendisine çeken tek yıllık bir çiçektir.");
                                }else if(res2[0].equalsIgnoreCase("rose")){
                                    textView.setText(res2[0].toUpperCase());
                                    textView2.setText("Gülgiller familyasının örnek bitkisidir. Pek çok gül türünün anayurdu Asya'dır. Ama gösterişli çiçekleri nedeniyle " +
                                            "neredeyse tüm dünyada yaygın şekilde yetiştirilmektedir. Türkiye'de yetişen 25 kadar yabani türü vardır.");
                                }else if(res2[0].equalsIgnoreCase("sunflower")){
                                    textView.setText(res2[0].toUpperCase());
                                    textView2.setText("Helianthus annuus tek yıllık bir ottur. Geçirgen ve nemli ya da kuru toprağı ve güneşli ya da yarı gölgeli bölgeleri tercih eder. Donlara ve kuraklığa dayanıklıdır.");
                                }else if(res2[0].equalsIgnoreCase("tulip")){
                                    textView.setText(res2[0].toUpperCase());
                                    textView2.setText("Lale, zambakgiller (Liliaceae) familyasından Tulipa cinsini oluşturan güzel çiçekleri ile süs bitkisi olarak yetiştirilen, soğanlı, çok yıllık otsu bitki türlerinin ortak adı.");
                                }else{
                                    textView.setText("BULUNAMADI");
                                }

                                Toast.makeText(MainActivity.this, "Bulundu", Toast.LENGTH_SHORT).show();
                            }

                            @Override
                            public void onFailure(Call<String> call, Throwable t) {
                                dialog.dismiss();
                                Toast.makeText(MainActivity.this, "" + t.getMessage(), Toast.LENGTH_SHORT).show();
                            }
                        });
                    }
                }).start();
            }
        }
        else {
            Toast.makeText(this,"dosyayi upload edemedim",Toast.LENGTH_SHORT).show();
        }
    }

    private void chooseFile() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent,PICK_FILE_REQUEST);

    }

    @Override
    public void onProgressUpdate(int percent) {

    }
}
