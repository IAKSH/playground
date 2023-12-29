package me.iaksh;

import org.lwjgl.openal.*;

public class TriangleWave extends Wave {

    private float amplitude;
    private float phaseShift;

    public TriangleWave(int durationMs) {
        super(durationMs);
        amplitude = 1.0f;
        phaseShift = 1.0f;
    }

    @Override
    public void generate(int sampleRate) {
        reallocAlBuffer();
        int samplesPerCycle = sampleRate / getFreq();
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
        AL11.alBufferData(getAlBuffer(), AL11.AL_FORMAT_MONO16, data, sampleRate);
    }

    public void setAmplitude(float amplitude) {
        if(amplitude < 0.0f || amplitude > 1.0f)
            throw new IllegalArgumentException();
        this.amplitude = amplitude;
    }

    public void setPhaseShift(float phaseShift) {
        if(phaseShift < 0.0f || phaseShift > 1.0f)
            throw new IllegalArgumentException();
        this.phaseShift = phaseShift;
    }

    public float getAmplitude() {
        return amplitude;
    }

    public float getPhaseShift() {
        return phaseShift;
    }
}
