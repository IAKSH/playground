package me.iaksh.mixer;

import me.iaksh.oscillator.Oscillator;
import me.iaksh.notation.Section;

import java.util.ArrayList;

public abstract class Track {
    protected final ArrayList<Short> waveform;
    protected final int bpm;

    public Track(int bpm) {
        waveform = new ArrayList<>();
        this.bpm = bpm;
    }

    public abstract ArrayList<Short> genWaveform(Oscillator cluster, ArrayList<Section> sections);
}
