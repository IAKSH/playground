package me.iaksh;

import org.lwjgl.openal.AL11;

public class Main {
    public static void main(String[] args) {
        Mixer m = new Mixer();
        m.play();
        m.destroy();
    }
}