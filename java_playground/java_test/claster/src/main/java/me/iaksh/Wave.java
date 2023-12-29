package me.iaksh;

import org.lwjgl.openal.*;

public abstract class Wave {
    private int alBuffer;
    private int durationMs;
    private int freq;

    protected void reallocAlBuffer() {
        AL11.alDeleteBuffers(alBuffer);
        alBuffer = AL11.alGenBuffers();
    }

    public Wave(int durationMs) {
        if(durationMs < 0) {
            throw new RuntimeException(String.format("durationMs must be positive, but given %d",durationMs));
        }
        this.durationMs = durationMs;
    }

    public int getAlBuffer() {
        return alBuffer;
    }

    public int getDurationMs() {
        return durationMs;
    }

    public int getFreq() {
        return freq;
    }

    public void setFreq(int freq) {
        if(freq <= 0) {
            throw new RuntimeException(String.format("freq must be positive, but given %d",freq));
        }
        this.freq = freq;
    }

    public abstract void generate(int sampleRate);
}
