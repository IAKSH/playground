package org.example;

import org.example.MyTest;
import org.example.FileReader;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        new MyTest().doIt();

        try {
            System.out.print(new FileReader("D:\\Programming-Env\\jdk-17.0.8+7\\NOTICE").read());
        } catch (IOException ioException) {
            throw new RuntimeException(ioException);
        }
    }
}