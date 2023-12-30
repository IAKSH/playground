package me.iaksh.mixer;

import me.iaksh.notation.Section;

import java.util.ArrayList;

// Mixer负责组织整首曲目
public interface Player {
    void play(ArrayList<ArrayList<Section>> sections);
}
