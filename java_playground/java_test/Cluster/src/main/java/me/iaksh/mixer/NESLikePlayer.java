package me.iaksh.mixer;

import me.iaksh.cluster.SquareWaveCluster;
import me.iaksh.cluster.TriangleWaveCluster;
import me.iaksh.cluster.WhiteNoiseCluster;
import me.iaksh.notation.Section;

import java.util.ArrayList;

public class NESLikePlayer implements Player {

    private ArrayList<Track> tracks;

    private void destroyTracks() {
        for(Track track : tracks)
            track.destroy();
    }

    private void waitAllTrackToBeFinished() {
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

    public NESLikePlayer(int bpm) {
        tracks = new ArrayList<>();
        for(int i = 0;i < 4;i++)
            tracks.add(new Track(bpm));
    }

    @Override
    public void play(ArrayList<ArrayList<Section>> sections) {
        if(sections.size() != 4)
            throw new IllegalArgumentException();

        tracks.get(0).genWaveform(new SquareWaveCluster(),sections.get(0));
        tracks.get(1).genWaveform(new SquareWaveCluster(),sections.get(1));
        tracks.get(2).genWaveform(new TriangleWaveCluster(),sections.get(2));
        tracks.get(3).genWaveform(new WhiteNoiseCluster(),sections.get(3));
        System.out.println("genWaveform() finished!");
        for(Track track : tracks)
            track.play();

        waitAllTrackToBeFinished();
        System.out.println("All track finished!");
        destroyTracks();
    }
}
