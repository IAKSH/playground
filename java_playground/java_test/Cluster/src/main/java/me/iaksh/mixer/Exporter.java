package me.iaksh.mixer;

import me.iaksh.notation.Section;

import java.util.ArrayList;

public interface Exporter {
    void saveToWav(String path,ArrayList<ArrayList<Section>> sections);
}
