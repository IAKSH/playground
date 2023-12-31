package me.iaksh.mixer;

import java.util.ArrayList;

public abstract class Synthesizer {
    protected ArrayList<Track> tracks;

    public Synthesizer() {
        tracks = new ArrayList<>();
    }

    protected void destroyTracks() {
        for(Track track : tracks)
            track.destroy();
    }

    protected void waitAllTrackToBeFinished() {
        try {
            boolean finished = false;
            while(!finished) {
                finished = true;
                for(Track track : tracks)
                    if(!track.isFinished())
                        finished = false;
                Thread.sleep(1);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
