package me.iaksh.cluster;

// Cluster只负责声音的听感
public interface Cluster {
    int sampleRate = 44100;

    short[] genWaveform(int ms,int simpleScore, int octaveShift, int semitoneShift);

    default int getSampleRate() {
        return sampleRate;
    }
}
