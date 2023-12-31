package me.iaksh.oscillator;

public class NoiseOscillator extends Oscillator {

    private short[] genBasicWaveform(int samplesPerCycle) {
        short[] data = new short[samplesPerCycle];

        for (int i = 0; i < data.length; i++) {
            data[i] = (short) (Math.random() * (Short.MAX_VALUE - Short.MIN_VALUE) + Short.MIN_VALUE);
        }
        return data;
    }

    @Override
    public short[] genWaveform(int ms,int simpleScore, int octaveShift, int semitoneShift) {
        short[] croppedData = new short[ms * getSampleRate() / 1000];
        if(simpleScore == 0) {
            return croppedData;
        }

        int samplesPerCycle = getSampleRate() / EqualTemp.toFreq(simpleScore,octaveShift,semitoneShift);
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
