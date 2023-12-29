package me.iaksh;

public class Note {
    private final int frequency;
    private final float gain;

    public Note(int frequency,float gain) {
        if(frequency < 0 || gain < 0)
            throw new IllegalArgumentException();
        this.frequency = frequency;
        this.gain = gain;
    }

    public int getFrequency() {
        return frequency;
    }

    public float getGain() {
        return gain;
    }
}
