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
        int[] simpleScores = {
                2,2,1,0,2,2,1,0,2,2,1,0,2,0,4,3,0,2,2,1,0,2,2,3,0,6,5,4,3,
                2,2,1,0,2,2,1,0,2,2,1,0,2,0,4,3,0,2,2,1,0,2,2,3,0,6,5,4,3,
                2,3,4,5,6,0,2,1,0,6,2,0,6,5,4,3,0,2,3,4,5,6,0,5,4,0,3,2,3,4,0,3,2,1,3,
                2,3,4,5,6,0,2,1,6,2,0,6,5,4,3,0,2,3,4,5,6,0,5,4,0,3,4,5,6,
                2,3,4,5,6,0,2,1,6,2,0,6,5,4,3,0,2,3,4,5,6,0,5,4,0,3,2,3,4,0,3,2,1,3,
                2,3,4,5,6,0,2,1,6,2,0,6,5,4,3,0,2,3,4,5,6,0,5,4,0,3,4,5,6,
                1,2,6,5,6,0,5,6,0,6,5,6,5,6,5,4,0,3,1,2,0,1,2,3,4,0,5,6,2,
                6,1,1,2,6,5,6,0,5,6,1,2,6,5,6,0,5,6,5,4,0,3,1,2,0,1,2,3,4,0,5,6,2,
                6,1,1,2,6,5,6,0,5,6,1,2,6,5,6,0,5,6,5,4,0,3,1,2,0,1,2,3,4,0,5,6,2,
                6,1,1,2,6,5,6,0,5,6,1,2,6,5,6,0,2,3,4,3,0,2,1,6,0,5,6,5,4,0,3,1,2,
                6,1,1,2,6,5,6,0,5,6,1,2,6,5,6,0,5,6,5,4,0,3,1,2,0,1,2,3,4,0,5,6,2,
                6,1,1,2,6,5,6,0,5,6,1,2,6,5,6,0,2,3,4,3,2,1,6,5,6,5,4,3,1,2,
                7,2,2,3,7,6,7,0,6,7,2,3,7,6,7,0,6,7,6,5,0,4,2,3,0,2,3,4,5,0,6,7,3,
                7,2,2,3,7,6,7,0,6,7,2,3,7,6,7,0,3,4,5,4,3,2,7,0,6,7,6,5,0,4,2,3,
                7,2,2,3,7,6,7,0,6,7,2,3,7,6,7,0,6,7,6,5,0,4,2,3,0,2,3,4,5,0,6,7,3,
                7,2,2,3,7,6,7,0,6,7,2,3,7,6,7,0,3,4,5,4,0,3,2,7,0,6,7,6,5,0,4,2,3
        };
        int[] frequencies = SimpleScoreTranslator.convert(simpleScores);
        for(int i = 0; i < 10;i++) {
            for(int j : frequencies) {
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
