package me.iaksh.notation;

import java.util.ArrayList;

public class Section {
    private final int timeSignature0;
    private final int timeSignature1;
    private ArrayList<Note> notes;

    public Section(int ts0,int ts1) {
        timeSignature0 = ts0;
        timeSignature1 = ts1;
        notes = new ArrayList<>();
    }

    public ArrayList<Note> getNotes() {
        return notes;
    }

    public int getTimeSignature0() {
        return timeSignature0;
    }

    public int getTimeSignature1() {
        return timeSignature1;
    }
}
