package me.iaksh.oscillator;

import java.util.Random;

public class NoiseOscillator extends CroppingOscillator {
    private final Random random;
    private float amplitude = 0.5f;

    @Override
    protected short[] genBasicWaveform(int samplesPerCycle) {
        short[] data = new short[samplesPerCycle];

        for (int i = 0; i < data.length; i++) {
            data[i] = (short) (random.nextGaussian() % amplitude * Short.MAX_VALUE);
        }
        return data;
    }

    public NoiseOscillator() {
        random = new Random();
    }

    public float getAmplitude() {
        return amplitude;
    }

    public void setAmplitude(float amplitude) {
        this.amplitude = amplitude;
    }
}
