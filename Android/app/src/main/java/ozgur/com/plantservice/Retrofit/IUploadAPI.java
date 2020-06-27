package ozgur.com.plantservice.Retrofit;

import java.io.File;

import okhttp3.MultipartBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface IUploadAPI {
    @Multipart
    @POST("api/upload")
    Call<String> uploadFile(@Part MultipartBody.Part File);
}
