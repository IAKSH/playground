package me.iaksh;

import org.lwjgl.openal.*;

public class SquareWave extends Wave {

    private float highRatio;

    public SquareWave(int durationMs) {
        super(durationMs);
        highRatio = 0.5f;
    }

    @Override
    public void generate(int sampleRate) {
        reallocAlBuffer();
        int samplesPerCycle =  sampleRate / getFreq();
        short[] data = new short[samplesPerCycle];
        for (int j = 0; j < data.length; j++) {
            if (j % samplesPerCycle < samplesPerCycle * highRatio) {
                data[j] = 0b000000000000000;
            } else {
                data[j] = 0b111111111111111;
            }
        }
        AL11.alBufferData(getAlBuffer(), AL11.AL_FORMAT_MONO16, data, sampleRate);
    }

    public void setHighRatio(float ratio) {
        if(ratio < 0.0f || ratio > 1.0f)
            throw new IllegalArgumentException();
        highRatio = ratio;
    }

    public float getHighRatio() {
        return highRatio;
    }
}
