package me.iaksh;

public abstract class Channel implements Runnable {

    protected Claster claster;

    public Channel() {
        claster = new Claster();
    }

    public void destoryClaster() {
        claster.destroyAlSource();
    }
}
