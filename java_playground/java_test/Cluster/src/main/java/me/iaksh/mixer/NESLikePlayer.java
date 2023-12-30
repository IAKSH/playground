package me.iaksh.mixer;

import me.iaksh.cluster.SquareWaveCluster;
import me.iaksh.cluster.TriangleWaveCluster;
import me.iaksh.cluster.WhiteNoiseCluster;
import me.iaksh.notation.Section;

import java.util.ArrayList;

public class NESLikePlayer implements Player {

    private ArrayList<Track> tracks;
    private TrackSynchronizer synchronizer;

    private void initTracks(int bpm) {
        Track sq0 = new Track(new SquareWaveCluster(44100),bpm);
        Track sq1 = new Track(new SquareWaveCluster(44100),bpm);
        Track tri = new Track(new TriangleWaveCluster(44100),bpm);
        Track noi = new Track(new WhiteNoiseCluster(44100),bpm);
        synchronizer.addTrack(sq0);
        synchronizer.addTrack(sq1);
        synchronizer.addTrack(tri);
        synchronizer.addTrack(noi);
        tracks.add(sq0);
        tracks.add(sq1);
        tracks.add(tri);
        tracks.add(noi);
    }

    public NESLikePlayer(int bpm) {
        tracks = new ArrayList<>();
        synchronizer = new TrackSynchronizer();
        initTracks(bpm);
    }

    @Override
    public void play(ArrayList<ArrayList<Section>> sections) {
        if(sections.size() != 4)
            throw new IllegalArgumentException();

        for(int i = 0;i < tracks.size();i++) {
            for(Section section : sections.get(i)) {
                tracks.get(i).addSection(section);
            }
        }

        for(Track track : tracks)
            track.start();
        while(!synchronizer.shouldExit()) {
            synchronizer.onTick();
        }
    }
}
