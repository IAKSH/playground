package me.iaksh.oscillator;

public class SquareOscillator extends Oscillator {
    private float dutyCycle = 0.5f;
    private float phaseShift = 1.0f;
    private float amplitude = 0.5f;

    private short[] genBasicWaveform(int samplesPerCycle) {
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

    @Override
    public short[] genWaveform(int ms,int freq) {
        short[] croppedData = new short[ms * getSampleRate() / 1000];
        if(freq == 0) {
            return croppedData;
        }

        int samplesPerCycle = getSampleRate() / freq;
        short[] data = genBasicWaveform(samplesPerCycle);

        if(croppedData.length > samplesPerCycle) {
            for(int i = 0;i < croppedData.length;i++) {
                croppedData[i] = data[i % data.length];
            }
        } else {
            System.arraycopy(data, 0, croppedData, 0, croppedData.length);
        }

        return croppedData;
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
