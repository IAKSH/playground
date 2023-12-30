package me.iaksh.cluster;

// Cluster只负责声音的听感
public interface Cluster {
    void play(int simpleScore,int octaveShift,int semitoneShift);
    void stop();
}
