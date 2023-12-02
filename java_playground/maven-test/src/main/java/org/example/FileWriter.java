package org.example;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;

public class FileWriter {
    protected OutputStream stream;

    public FileWriter(String path) throws FileNotFoundException {
        stream = new FileOutputStream(path);
    }

    public void write(String str) {

    }

    public void appen(String str) {

    }
}
