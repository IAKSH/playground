package me.iaksh;

public class Note {
    private final int simpleScore;
    private final int octaveShift;
    private final int semitoneShift;
    private final float gain;
    private final float noteFraction;

    public Note(float noteFraction,int octaveShift,int semitoneShift,int simpleScore,float gain) {
        if(noteFraction == 0.0f || simpleScore < 0 || simpleScore > 7 || gain < 0)
            throw new IllegalArgumentException();
        this.noteFraction = noteFraction;
        this.octaveShift = octaveShift;
        this.semitoneShift = semitoneShift;
        this.simpleScore = simpleScore;
        this.gain = gain;
    }

    public int getSimpleScore() {
        return simpleScore;
    }

    public float getGain() {
        return gain;
    }

    public float getNoteFraction() {
        return noteFraction;
    }

    public int getOctaveShift() {
        return octaveShift;
    }

    public int getSemitoneShift() {
        return semitoneShift;
    }
}