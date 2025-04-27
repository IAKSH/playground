package org.example;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

public class FileReader {
    protected InputStream stream;
    public FileReader(String path) throws FileNotFoundException {
        stream = new FileInputStream(path);
    }

    public String read() throws IOException {
        StringBuilder buffer = new StringBuilder();
        int len = stream.available();
        for(int i = 0;i < len;i++) {
            buffer.append((char) stream.read());
        }

        buffer.append("\r\n");
        return buffer.toString();
    }
}
