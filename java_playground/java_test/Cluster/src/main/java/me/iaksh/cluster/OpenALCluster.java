package me.iaksh.cluster;

public abstract class OpenALCluster implements Cluster {
    protected OpenALSource source;
    protected OpenALBuffer buffer;

    public  OpenALCluster() {
        source = new OpenALSource();
        buffer = new OpenALBuffer();
    }

    public void start() {
        source.play(buffer);
    }

    public void destroy() {
        source.destroyALSource();
        buffer.destroyOpenALBuffer();
    }

    @Override
    public void stop() {
        source.stop();
    }
}
