package me.iaksh.notation;

public class Note {
    private final int simpleScore;
    private final int octaveShift;
    private final int semitoneShift;
    private final float noteFraction;
    private final boolean dotted;

    public Note(int score,int oct,int semi,float fra,boolean dot) {
        simpleScore = score;
        octaveShift = oct;
        semitoneShift = semi;
        noteFraction = fra;
        dotted = dot;
    }

    public int getSemitoneShift() {
        return semitoneShift;
    }

    public int getOctaveShift() {
        return octaveShift;
    }

    public int getSimpleScore() {
        return simpleScore;
    }

    public float getNoteFraction() {
        return noteFraction;
    }

    public boolean isDotted() {
        return dotted;
    }
}
