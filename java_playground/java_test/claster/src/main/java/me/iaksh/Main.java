package me.iaksh;

import org.lwjgl.openal.AL11;

public class Main {
    private Mixer mixer;

    void initMixer() {
        mixer = new Mixer();
    }

    void destroy() {
        mixer.destroy();
    }

    void start() throws InterruptedException {
        initMixer();
        mixer.play();
        destroy();
    }

    public static void main(String[] args) {
        try {
            Main m = new Main();
            m.start();
            m.destroy();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}