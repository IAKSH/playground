package me.iaksh.mixer;

import me.iaksh.cluster.Cluster;
import me.iaksh.notation.Note;
import me.iaksh.notation.Section;
import org.lwjgl.openal.AL11;

import java.util.ArrayList;

public class Track {
    private final int alSource;
    private final int alBuffer;
    private final ArrayList<Short> waveform;
    private final int bpm;

    private void loadAlBuffer(int sampleRate) {
        short[] buffer = new short[waveform.size()];
        for(int i = 0;i < buffer.length;i++)
            buffer[i] = waveform.get(i);
        AL11.alBufferData(alBuffer,AL11.AL_FORMAT_MONO16,buffer,sampleRate);
    }

    public Track(int bpm) {
        alSource = AL11.alGenSources();
        alBuffer = AL11.alGenBuffers();
        waveform = new ArrayList<>();
        this.bpm = bpm;

        // temp
        AL11.alSourcef(alSource,AL11.AL_GAIN,0.05f);
    }

    public void destroy() {
        AL11.alDeleteSources(alSource);
        AL11.alDeleteBuffers(alBuffer);
    }

    public ArrayList<Short> genWaveform(Cluster cluster, ArrayList<Section> sections) {
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

    public void play() {
        AL11.alSourceStop(alSource);
        AL11.alSourcei(alSource,AL11.AL_BUFFER,alBuffer);
        AL11.alSourcePlay(alSource);
    }

    public void stop() {
        AL11.alSourceStop(alSource);
    }

    public void enableLoop() {
        AL11.alSourcei(alSource,AL11.AL_LOOPING,1);
    }

    public void disableLoop() {
        AL11.alSourcei(alSource,AL11.AL_LOOPING,0);
    }

    public boolean isFinished() {
        return AL11.alGetSourcei(alSource,AL11.AL_SOURCE_STATE) == AL11.AL_STOPPED;
    }
}
