package me.iaksh.mixer;

import me.iaksh.notation.Section;

import java.util.ArrayList;

public interface MusicPlayer {
    void play(ArrayList<ArrayList<Section>> sections);
    void play(float gain,ArrayList<ArrayList<Section>> sections);
}
