package me.iaksh;

import org.lwjgl.openal.AL11;

public class Channel {
    private int alSource;
    private int bindingAlBuffer;

    public Channel() {
        alSource = AL11.alGenSources();
        AL11.alSourcei(alSource,AL11.AL_LOOPING,1);
    }

    public void bindBuffer(int alBuffer) {
        AL11.alSourceStop(alSource);
        AL11.alSourcei(alSource,AL11.AL_BUFFER,alBuffer);
        bindingAlBuffer = alBuffer;
    }

    public void play() {
        AL11.alSourcePlay(alSource);
        //AL11.alDeleteBuffers(bindingAlBuffer);
    }

    public void stop() {
        AL11.alSourceStop(alSource);
    }

    public void setGain(float gain) {
        if(gain < 0.0f)
            throw new IllegalArgumentException();
        AL11.alSourcef(alSource,AL11.AL_GAIN,gain);
    }

    public void destroyAlSource() {
        AL11.alDeleteSources(alSource);
    }
}