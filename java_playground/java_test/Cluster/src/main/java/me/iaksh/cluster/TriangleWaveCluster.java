package me.iaksh.cluster;

public class TriangleWaveCluster extends OpenALCluster {
    private final int sampleRate;
    private final float amplitude = 1.0f;
    private final float phaseShift = 1.0f;

    public TriangleWaveCluster(int sampleRate) {
        this.sampleRate = sampleRate;
    }

    @Override
    public void play(int simpleScore, int octaveShift, int semitoneShift) {
        int samplesPerCycle = (int) (sampleRate / EqualTemp.toFreq(simpleScore,octaveShift,semitoneShift));
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

        buffer.write(data,sampleRate);
        source.play(buffer);
    }
}
