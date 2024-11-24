package com.ysc.utils;


import com.qcloud.cos.COSClient;
import com.qcloud.cos.ClientConfig;
import com.qcloud.cos.auth.BasicCOSCredentials;
import com.qcloud.cos.auth.COSCredentials;
import com.qcloud.cos.http.HttpProtocol;
import com.qcloud.cos.model.ObjectMetadata;
import com.qcloud.cos.model.PutObjectResult;
import com.qcloud.cos.region.Region;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.net.URL;
import java.util.UUID;
//交给IOC容器管理
@Component
public class TencentCOSUtil {

    // COS的SecretId
    private static final String secretId = "?";
    // COS的SecretKey
    private static final String secretKey = "?";
    //文件上传后访问路径的根路径，后面要最佳文件名字与类型
    private static final String rootSrc = "?";
    //上传的存储桶的地域，可参考根路径https://qq-test-1303******.cos.地域.myqcloud.com,此参数在COS的后台能查询。
    private static final String bucketAddr = "?";
    //存储桶的名字，是自己在存储空间自己创建的，我创建的名字是：qq-test-1303******
    private static final String bucketName = "?";

    /**
     * 1.调用静态方法getCosClient()就会获得COSClient实例
     * 2.本方法根据永久密钥初始化 COSClient的，官方是不推荐，官方推荐使用临时密钥，是可以限制密钥使用权限，创建cred时有些区别
     *
     * @return COSClient实例
     */
    private static COSClient getCosClient() {
        // 1 初始化用户身份信息（secretId, secretKey）。
        COSCredentials cred = new BasicCOSCredentials(secretId, secretKey);
        // 2.1 设置存储桶的地域（上文获得）
        Region region = new Region(bucketAddr);
        ClientConfig clientConfig = new ClientConfig(region);
        // 2.2 使用https协议传输
        clientConfig.setHttpProtocol(HttpProtocol.https);
        // 3 生成 cos 客户端。
        // 返回COS客户端
        return new COSClient(cred, clientConfig);
    }

    /**
     * 只要调用静态方法upLoadFile(MultipartFile multipartFile)就可以获取上传后文件的全路径
     *
     * @param file
     * @return 返回文件的浏览全路径
     */
    public static String upLoadFile(MultipartFile file) {
        try {
            // 获取上传的文件的输入流
            InputStream inputStream = file.getInputStream();

            // 避免文件覆盖，获取文件的原始名称，如123.jpg,然后通过截取获得文件的后缀，也就是文件的类型
            String originalFilename = file.getOriginalFilename();
            //获取文件的类型
            String fileType = null;
            if (originalFilename != null) {
                fileType = originalFilename.substring(originalFilename.lastIndexOf("."));
            }
            //使用UUID工具  创建唯一名称，放置文件重名被覆盖，在拼接上上命令获取的文件类型
            String fileName = UUID.randomUUID().toString() + fileType;
            // 指定文件上传到 COS 上的路径，即对象键。最终文件会传到存储桶名字中的images文件夹下的fileName名字
            String key = "/avatar/" + fileName;
            // 创建上传Object的Metadata
            ObjectMetadata objectMetadata = new ObjectMetadata();
            // - 使用输入流存储，需要设置请求长度
            objectMetadata.setContentLength(inputStream.available());
            // - 设置缓存
            objectMetadata.setCacheControl("no-cache");
            // - 设置Content-Type
            objectMetadata.setContentType(fileType);
            //上传文件
            PutObjectResult putResult = getCosClient().putObject(bucketName, key, inputStream, objectMetadata);
            // 创建文件的网络访问路径
            String url = rootSrc + key;
            //关闭 cosClient，并释放 HTTP 连接的后台管理线程
            getCosClient().shutdown();
            return url;

        } catch (Exception e) {
            e.printStackTrace();
            // 发生IO异常、COS连接异常等，返回空
            return null;
        }
    }

}