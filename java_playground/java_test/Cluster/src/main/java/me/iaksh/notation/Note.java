package me.iaksh.notation;

public abstract class Note {
    protected float noteFraction;
    protected boolean dotted;

    public Note(float noteFraction,boolean dotted) {
        this.noteFraction = noteFraction;
        this.dotted = dotted;
    }

    public float getNoteFraction() {
        return noteFraction;
    }

    public boolean isDotted() {
        return dotted;
    }

    public abstract int getFreq();
}
