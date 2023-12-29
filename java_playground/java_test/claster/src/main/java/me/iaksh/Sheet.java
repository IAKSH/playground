package me.iaksh;

import java.util.ArrayList;

public class Sheet {
    private int bpm;
    private int currentIndex;
    private ArrayList<Frame> frames;

    private void init() {
        frames = new ArrayList<>();
    }

    public Sheet(String path) {
        init();
        // TODO: load from json
    }

    // Test only
    public Sheet() {
        init();
        bpm = 400;
        for(int i = 0;i < 100;i++) {
            frames.add(new Frame(
                    new Note(500 + (i % 5) * (i % 3) * 100,0.07f),
                    new Note(200 + (i % 3) * 30,0.1f),
                    new Note(500 + (i % 7) * ((i % 2 == 1) ? 10 : -10),0.2f),
                    new Note((i % 2 == 1) ? 0 : 1,0.1f)
            ));
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
