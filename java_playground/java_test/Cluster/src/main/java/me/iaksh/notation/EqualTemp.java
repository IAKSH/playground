package me.iaksh.notation;

public class EqualTemp {
    public static int toFreq(int simpleScore,int octaveShift,int semitoneShift) {
        double[] equalTemperaments = {1,261.63,293.66,329.63,349.23,392.00,440.00,493.88};
        double baseFreq = equalTemperaments[simpleScore];
        double freq = baseFreq *
                Math.pow(2, octaveShift) *
                Math.pow(2, semitoneShift / 12.0);
        return (int) freq;
    }
}
