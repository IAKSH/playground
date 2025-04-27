package me.iaksh.notation;

public class EqualTempNote extends Note {
    private final int simpleScore;
    private final int octaveShift;
    private final int semitoneShift;

    public int toFreq(int simpleScore,int octaveShift,int semitoneShift) {
        double[] equalTemperaments = {0,261.63,293.66,329.63,349.23,392.00,440.00,493.88};
        double baseFreq = equalTemperaments[simpleScore];
        double freq = baseFreq *
                Math.pow(2, octaveShift) *
                Math.pow(2, semitoneShift / 12.0);
        return (int) freq;
    }

    public EqualTempNote(float fra, boolean dot,int score, int oct, int semi) {
        super(fra,dot);
        simpleScore = score;
        octaveShift = oct;
        semitoneShift = semi;
    }

    @Override
    public int getFreq() {
        return toFreq(simpleScore,octaveShift,semitoneShift);
    }
}
