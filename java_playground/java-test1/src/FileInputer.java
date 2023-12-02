import java.io.*;

public class FileInputer {
    private InputStream istream;
    public FileInputer(String path) {
        try {
            istream = new FileInputStream(path);
        } catch(FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public String readAll() {
        StringBuilder strBuilder = new StringBuilder();
        try {
            var len = istream.available();
            for(int i = 0;i < len;i++) {
                strBuilder.append(istream.read());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return strBuilder.toString();
    }

    public char readChar() {
        char inputChar = 0;
        try {
            inputChar = (char) istream.read();
        } catch(IOException e) {
            e.printStackTrace();
        }

        return inputChar;
    }
}
