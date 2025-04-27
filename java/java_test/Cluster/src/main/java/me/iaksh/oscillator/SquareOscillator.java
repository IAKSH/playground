package me.iaksh.oscillator;

public class SquareOscillator extends CroppingOscillator {
    private float dutyCycle = 0.5f;
    private float phaseShift = 1.0f;
    private float amplitude = 0.5f;

    @Override
    protected short[] genBasicWaveform(int samplesPerCycle) {
        short[] data = new short[samplesPerCycle];
        int halfSamples = (int) (samplesPerCycle * dutyCycle);
        int phaseSamples = (int) (samplesPerCycle * phaseShift);

        for (int i = 0; i < samplesPerCycle; i++) {
            if ((i + phaseSamples) % samplesPerCycle < halfSamples) {
                data[i] = (short) (Short.MAX_VALUE * amplitude);
            } else {
                data[i] = (short) (Short.MIN_VALUE * amplitude);
            }
        }
        return data;
    }

    public float getAmplitude() {
        return amplitude;
    }

    public float getDutyCycle() {
        return dutyCycle;
    }

    public float getPhaseShift() {
        return phaseShift;
    }

    public void setAmplitude(float amplitude) {
        this.amplitude = amplitude;
    }

    public void setDutyCycle(float dutyCycle) {
        this.dutyCycle = dutyCycle;
    }

    public void setPhaseShift(float phaseShift) {
        this.phaseShift = phaseShift;
    }
}
