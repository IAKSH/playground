package me.iaksh.oscillator;

import me.iaksh.mixer.EqualTemp;

public class TriangleOscillator extends Oscillator {
    private final float amplitude = 1.0f;
    private final float phaseShift = 1.0f;

    private short[] genBasicWaveform(int samplesPerCycle) {
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
