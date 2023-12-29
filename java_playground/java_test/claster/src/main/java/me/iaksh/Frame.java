package me.iaksh;

public class Frame {

    private final Note[] notes;

    public Frame(Note sq0,Note sq1,Note tri,Note noise) {
        notes = new Note[]{sq0, sq1, tri, noise};
    }

    public Note getSq0() {
        return notes[0];
    }

    public Note getSq1() {
        return notes[1];
    }

    public Note getTri() {
        return notes[2];
    }

    public Note getNoise() {
        return notes[3];
    }
}
