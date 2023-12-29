package me.iaksh;

import org.lwjgl.openal.*;

import java.util.Random;

public class NoiseWave extends Wave{

    private float amplitude;
    private float noiseFactor;

    public NoiseWave(int durationMs) {
        super(durationMs);
        amplitude = 1.0f;
        noiseFactor = 1.0f;
    }

    @Override
    public void generate(int sampleRate) {
        // TODO: 严重的内存问题
        // 会分配大量内存且无法gc
        final int duration = 1;

        reallocAlBuffer();
        int numSamples = (int) (sampleRate * duration);
        short[] data = new short[numSamples];
        float maxAmplitude = Short.MAX_VALUE * amplitude;
        Random random = new Random();

        for (int i = 0; i < numSamples; i++) {
            float randomValue = (float) (random.nextFloat() * 2 - 1); // Generate a value between -1 and 1
            float noise = randomValue * noiseFactor * maxAmplitude;
            data[i] = (short) noise;
        }

        float repeatRate = duration * getFreq();
        int repeatSamples = (int) (repeatRate * sampleRate);

        short[] repeatedData = new short[repeatSamples];
        for (int i = 0; i < repeatSamples; i++) {
            repeatedData[i] = data[i % numSamples];
        }

        AL11.alBufferData(getAlBuffer(), AL11.AL_FORMAT_MONO16, repeatedData, sampleRate);
    }

    public void setAmplitude(float amplitude) {
        if(amplitude < 0.0f || amplitude > 1.0f)
            throw new IllegalArgumentException();
        this.amplitude = amplitude;
    }

    public void setNoiseFactor(float noiseFactor) {
        this.noiseFactor = noiseFactor;
    }

    public float getAmplitude() {
        return amplitude;
    }

    public float getNoiseFactor() {
        return noiseFactor;
    }
}
