import java.io.*;
import java.nio.charset.StandardCharsets;

public class FileWriter {
    private OutputStreamWriter writer;

    public FileWriter(String path) {
        try {
            OutputStream ostream = new FileOutputStream(path);
            writer = new OutputStreamWriter(ostream, StandardCharsets.UTF_8);
        } catch(FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    void writeChar(char outputChar) {
        try {
            writer.append(outputChar);
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    void writeString(String outStr) {
        try {
            for(int i = 0;i < outStr.length();i++) {
                writer.append(outStr.charAt(i));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    void flush() {
        try {
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
