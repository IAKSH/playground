import java.io.BufferedReader;
import java.io.InputStreamReader;

public class ConsoleInputer {
    private final BufferedReader br;

    public ConsoleInputer() {
        br = new BufferedReader(new InputStreamReader(System.in));
    }
    char readChar() {
        char inputChar = 0;
        try {
            inputChar = (char)br.read();
        } catch(Exception e) {
            e.printStackTrace();
        }
        return inputChar;
    }

    String readLine() {
        String input = "";
        try {
            input = br.readLine();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return input;
    }
}
