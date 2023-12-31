package me.iaksh.mixer;

import me.iaksh.notation.Note;
import me.iaksh.notation.Section;
import me.iaksh.oscillator.Oscillator;

import java.util.ArrayList;

public class EqualTempTrack extends Track {

    public EqualTempTrack(int bpm) {
        super(bpm);
    }

    @Override
    public ArrayList<Short> genWaveform(Oscillator cluster, ArrayList<Section> sections){
        waveform.clear();
        for(Section section : sections) {
            for(Note note : section.getNotes()) {
                int durationMs = (int) (note.getNoteFraction() * section.getTimeSignature1() * 60000.0f / bpm);
                if(note.isDotted())
                    durationMs = (int)(durationMs * 1.5f);
                for(Short s : cluster.genWaveform(durationMs,EqualTemp.toFreq(note.getSimpleScore(),note.getOctaveShift(),note.getSemitoneShift())))
                    waveform.add(s);
            }
        }
        return waveform;
    }
}
