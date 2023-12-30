package me.iaksh.cluster;

public class SquareWaveCluster extends OpenALCluster {
    private final int sampleRate;
    private final float dutyCycle = 0.5f;
    private final float phaseShift = 1.0f;

    public SquareWaveCluster(int sampleRate) {
        this.sampleRate = sampleRate;
    }

    @Override
    public void play(int simpleScore, int octaveShift, int semitoneShift) {
        int samplesPerCycle = (int) (sampleRate / EqualTemp.toFreq(simpleScore,octaveShift,semitoneShift));
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

        buffer.write(data,sampleRate);
        source.play(buffer);
    }
}
