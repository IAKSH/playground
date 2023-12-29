package me.iaksh;

import org.lwjgl.openal.AL11;

// 很明显，音色由Cluster决定
// 想要多种音色，要么是Cluster提供更多API，要么就是提供更多Cluster
public class Cluster {

    private final int sampleRate;

    public Cluster(int sampleRate) {
        if (sampleRate <= 0)
            throw new IllegalArgumentException();
        this.sampleRate = sampleRate;
    }

    public int genSquare(int frequency) {
        return genSquare(frequency,0.5f,1.0f);
    }

    public int genSquare(int frequency, double dutyCycle, double phaseShift) {
        int alBuffer = AL11.alGenBuffers();
        if(frequency == 0) {
            return alBuffer;
        }

        int samplesPerCycle = (int) (sampleRate / frequency);
        short[] data = new short[samplesPerCycle];

        int halfSamples = (int) (samplesPerCycle * dutyCycle);
        int phaseSamples = (int) (samplesPerCycle * phaseShift);

        for (int i = 0; i < samplesPerCycle; i++) {
            if ((i + phaseSamples) % samplesPerCycle < halfSamples) {
                data[i] = Short.MAX_VALUE;
            } else {
                data[i] = Short.MIN_VALUE;
            }
        }

        AL11.alBufferData(alBuffer, AL11.AL_FORMAT_MONO16, data, sampleRate);
        return alBuffer;
    }

    public int genTriangle(int frequency) {
        return genTriangle(frequency,1.0f,1.0f);
    }

    public int genTriangle(int frequency,float amplitude, float phaseShift) {
        int alBuffer = AL11.alGenBuffers();
        if(frequency == 0) {
            return alBuffer;
        }

        int samplesPerCycle = (int) (sampleRate / frequency);
        short[] data = new short[samplesPerCycle];

        float maxAmplitude = Short.MAX_VALUE * amplitude;
        float phaseIncrement = (2 * (float) Math.PI) / samplesPerCycle;
        float currentPhase = 0;

        for (int j = 0; j < data.length; j++) {
            float value = (float) Math.sin(currentPhase);
            data[j] = (short) (value * maxAmplitude);

            currentPhase += phaseShift * phaseIncrement;
            if (currentPhase >= 2 * Math.PI) {
                currentPhase -= 2 * Math.PI;
            }
        }

        AL11.alBufferData(alBuffer, AL11.AL_FORMAT_MONO16, data, sampleRate);
        return alBuffer;
    }

    public int genWhiteNoise(float frequency) {
        int alBuffer = AL11.alGenBuffers();
        if(frequency == 0) {
            return alBuffer;
        }

        int samplesPerCycle = (int) (sampleRate / frequency);
        short[] data = new short[samplesPerCycle];

        for (int i = 0; i < data.length; i++) {
            data[i] = (short) (Math.random() * (Short.MAX_VALUE - Short.MIN_VALUE) + Short.MIN_VALUE);
        }

        AL11.alBufferData(alBuffer, AL11.AL_FORMAT_MONO16, data, sampleRate);
        return alBuffer;
    }
}