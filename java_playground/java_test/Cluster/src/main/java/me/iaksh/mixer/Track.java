package me.iaksh.mixer;

import me.iaksh.cluster.Cluster;
import me.iaksh.notation.Note;
import me.iaksh.notation.Section;

import java.util.ArrayList;

public class Track extends Thread {
    private Cluster cluster;
    private ArrayList<Section> sections;
    private final int bpm;
    private long startTimestamp;
    private boolean ready;

    public Track(Cluster cluster,int bpm) {
        this.cluster = cluster;
        this.bpm = bpm;
        ready = true;
        sections = new ArrayList<>();
    }

    @Override
    public void run() {
        try {
            for(Section section : sections) {
                for(Note note : section.getNotes()) {
                    cluster.play(note.getSimpleScore(),note.getOctaveShift(),note.getSemitoneShift());

                    startTimestamp = System.currentTimeMillis();
                    if(!note.isDotted()) {
                        while((System.currentTimeMillis() - startTimestamp) <
                                (int) (note.getNoteFraction() * section.getTimeSignature1() * 60000.0f / bpm));
                    } else {
                        while((System.currentTimeMillis() - startTimestamp) <
                                (int) (note.getNoteFraction() * section.getTimeSignature1() * 60000.0f / bpm * 1.5f));
                    }

                    cluster.stop();
                }
                ready = true;
                while(ready)
                    Thread.sleep(1);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void addSection(Section section) {
        sections.add(section);
    }

    public boolean isReady() {
        return ready;
    }

    public void goOn() {
        ready = false;
    }
}
