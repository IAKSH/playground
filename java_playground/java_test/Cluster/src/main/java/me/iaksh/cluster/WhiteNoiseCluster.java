package me.iaksh.cluster;

public class WhiteNoiseCluster extends OpenALCluster {

    private final int sampleRate;

    public WhiteNoiseCluster(int sampleRate) {
        this.sampleRate = sampleRate;
    }

    @Override
    public void play(int simpleScore, int octaveShift, int semitoneShift) {
        if(simpleScore == 0) {
            source.stop();
            return;
        }

        int samplesPerCycle = (int) (sampleRate / EqualTemp.toFreq(simpleScore,octaveShift,semitoneShift));
        short[] data = new short[samplesPerCycle];

        for (int i = 0; i < data.length; i++) {
            data[i] = (short) (Math.random() * (Short.MAX_VALUE - Short.MIN_VALUE) + Short.MIN_VALUE);
        }

        buffer.write(data,sampleRate);
        source.play(buffer);
    }
}
