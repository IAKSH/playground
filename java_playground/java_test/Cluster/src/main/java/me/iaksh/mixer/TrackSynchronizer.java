package me.iaksh.mixer;

import java.util.ArrayList;

public class TrackSynchronizer {
    private ArrayList<Track> tracks;

    private boolean allTrackReady() {
        for(Track track : tracks) {
            if(!track.isReady())
                return false;
        }
        return true;
    }

    private void runAllTrack() {
        for(Track track : tracks)
            track.goOn();
    }

    public TrackSynchronizer() {
        tracks = new ArrayList<>();
    }

    public void addTrack(Track track) {
        tracks.add(track);
    }

    public void onTick() {
        if(allTrackReady())
            runAllTrack();
    }

    public boolean shouldExit() {
        for(Track track : tracks)
            if(track.isAlive())
                return false;
        return true;
    }
}
