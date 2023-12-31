package me.iaksh.oscillator;

public abstract class Oscillator {

    public abstract short[] genWaveform(int ms,int simpleScore, int octaveShift, int semitoneShift);

    public static int getSampleRate() {
        return 44100;
    }
}
