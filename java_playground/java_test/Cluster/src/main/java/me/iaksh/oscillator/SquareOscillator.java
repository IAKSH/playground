package me.iaksh.oscillator;

import me.iaksh.mixer.EqualTemp;

public class SquareOscillator extends Oscillator {
    private final float dutyCycle = 0.5f;
    private final float phaseShift = 1.0f;

    private short[] genBasicWaveform(int samplesPerCycle) {
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
}
