package me.iaksh.oscillator;

import java.util.Random;

public class NoiseOscillator extends Oscillator {
    private final Random random;

    private short[] genBasicWaveform(int samplesPerCycle) {
        short[] data = new short[samplesPerCycle];

        for (int i = 0; i < data.length; i++) {
            data[i] = (short) (random.nextGaussian() * Short.MAX_VALUE);
        }
        return data;
    }

    public NoiseOscillator() {
        random = new Random();
    }

    @Override
    public short[] genWaveform(int ms,int freq) {
        short[] croppedData = new short[ms * getSampleRate() / 1000];
        if(freq == 0) {
            return croppedData;
        }

        int samplesPerCycle = getSampleRate() / freq * 200;
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
