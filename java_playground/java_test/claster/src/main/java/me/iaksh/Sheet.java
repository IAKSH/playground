package me.iaksh;

import java.util.ArrayList;

public class Sheet {
    private int bpm;
    private int currentIndex;
    private ArrayList<Frame> frames;

    private void init() {
        frames = new ArrayList<>();
    }

    private void loadFromJson(String json) {
        // TODO
    }

    private void loadFromFile(String path) {
        // TODO
    }

    public Sheet(String path) {
        init();
        loadFromFile(path);
    }

    // Test only
    public Sheet() {
        init();
        bpm = 240;
        //int[] a = {440, 293, 329, 392, 329, 293, 392, 329, 392, 493};
        int[] a = {440, 0, 293, 0, 329, 0, 392, 0, 329, 0, 293, 0, 392, 0, 329, 0, 392, 0, 493, 493};
        for(int i = 0; i < 10;i++) {
            for(int j : a) {
                frames.add(new Frame(
                        new Note(j,0.05f),
                        new Note(j * 2,0.05f),
                        new Note(j / 2,0.05f),
                        new Note(j % 2 == 1 ? 0 : 1,0.01f)
                ));
            }
        }
    }

    public void rewind() {
        currentIndex = 0;
    }

    public boolean eof() {
        return currentIndex >= frames.size();
    }

    public int getBpm() {
        return bpm;
    }

    public Frame nextFrame() {
        return frames.get(currentIndex++);
    }
}
