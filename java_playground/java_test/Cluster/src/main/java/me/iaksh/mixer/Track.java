package me.iaksh.mixer;

import me.iaksh.oscillator.Oscillator;
import me.iaksh.notation.Note;
import me.iaksh.notation.Section;

import java.util.ArrayList;

public class Track {
    private final ArrayList<Short> waveform;
    private final int bpm;

    private void loadAlBuffer(int sampleRate) {
        short[] buffer = new short[waveform.size()];
        for(int i = 0;i < buffer.length;i++)
            buffer[i] = waveform.get(i);
    }

    public Track(int bpm) {
        waveform = new ArrayList<>();
        this.bpm = bpm;
    }

    public ArrayList<Short> genWaveform(Oscillator cluster, ArrayList<Section> sections) {
        waveform.clear();
        for(Section section : sections) {
            for(Note note : section.getNotes()) {
                int durationMs = (int) (note.getNoteFraction() * section.getTimeSignature1() * 60000.0f / bpm);
                if(note.isDotted())
                    durationMs = (int)(durationMs * 1.5f);
                for(Short s : cluster.genWaveform(durationMs,note.getSimpleScore(),note.getOctaveShift(),note.getSemitoneShift()))
                    waveform.add(s);
            }
        }
        loadAlBuffer(cluster.getSampleRate());
        return waveform;
    }
}
