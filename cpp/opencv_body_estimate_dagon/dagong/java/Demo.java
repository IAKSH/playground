import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;

public class Demo {
    static { 
        System.loadLibrary("dagong");
    }
    
    static native String demo(String base64Img);
    
    public static void main(String args[]) throws Exception {
        byte[] fileContent = Files.readAllBytes(Paths.get("./a.png"));
        String encodedString = Base64.getEncoder().encodeToString(fileContent);
        String result = demo(encodedString);
        System.out.println("java received: " + result);
    }
}
