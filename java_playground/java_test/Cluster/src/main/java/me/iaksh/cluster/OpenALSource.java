package me.iaksh.cluster;

import org.lwjgl.openal.AL11;

public class OpenALSource {
    private int source;

    public OpenALSource() {
        source = AL11.alGenSources();
        AL11.alSourcei(source,AL11.AL_LOOPING,1);
        // temp
        AL11.alSourcef(source,AL11.AL_GAIN,0.05f);
    }

    public void destroyALSource() {
        AL11.alDeleteSources(source);
    }

    public void play(OpenALBuffer buffer) {
        AL11.alSourceStop(source);
        AL11.alSourcei(source,AL11.AL_BUFFER,buffer.getBufferID());
        AL11.alSourcePlay(source);
    }

    public void stop() {
        AL11.alSourceStop(source);
    }
}
