package me.iaksh.oscillator;

public abstract class Effect implements WaveGenerator {
    protected final Oscillator oscillator;
    public Effect(Oscillator oscillator) {
        this.oscillator = oscillator;
    }
}
