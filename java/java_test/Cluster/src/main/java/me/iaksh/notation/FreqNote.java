package me.iaksh.notation;

public class FreqNote extends Note {

    private int frequency;

    public FreqNote(float noteFraction, boolean dotted,int frequency) {
        super(noteFraction, dotted);
        this.frequency = frequency;
    }

    @Override
    public int getFreq() {
        return frequency;
    }
}
