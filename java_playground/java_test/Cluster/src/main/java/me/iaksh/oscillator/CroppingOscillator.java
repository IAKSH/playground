package me.iaksh.oscillator;

public abstract class CroppingOscillator extends Oscillator {
    protected abstract short[] genBasicWaveform(int samplesPerCycle);

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
