package me.iaksh.notation;

public class EqualTempNote implements Note {
    private final int simpleScore;
    private final int octaveShift;
    private final int semitoneShift;
    private final float noteFraction;
    private final boolean dotted;

    public EqualTempNote(int score, int oct, int semi, float fra, boolean dot) {
        simpleScore = score;
        octaveShift = oct;
        semitoneShift = semi;
        noteFraction = fra;
        dotted = dot;
    }

    @Override
    public float getNoteFraction() {
        return noteFraction;
    }

    @Override
    public boolean isDotted() {
        return dotted;
    }

    @Override
    public int getFreq() {
        return EqualTemp.toFreq(simpleScore,octaveShift,semitoneShift);
    }
}
