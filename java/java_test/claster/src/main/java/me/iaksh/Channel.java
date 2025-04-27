package me.iaksh;

import org.lwjgl.openal.AL11;

public class Channel {
    private int alSource;
    private int durationMs;
    private long startTimestamp;
    private boolean active;

    public Channel() {
        alSource = AL11.alGenSources();
        AL11.alSourcei(alSource,AL11.AL_LOOPING,1);
        active = false;
    }

    public void bindBuffer(int alBuffer) {
        AL11.alSourceStop(alSource);
        AL11.alSourcei(alSource,AL11.AL_BUFFER,alBuffer);
    }

    public void setDurationMs(int durationMs) {
        if(durationMs < 0)
            throw new IllegalArgumentException();
        this.durationMs = durationMs;
    }

    public void play() {
        startTimestamp = System.currentTimeMillis();
        AL11.alSourcePlay(alSource);
        active = true;
    }

    public void tryStop() {
        if(System.currentTimeMillis() - startTimestamp >= durationMs)
            stop();
    }

    public void stop() {
        AL11.alSourceStop(alSource);
        active = false;
    }

    public boolean isActive() {
        return active;
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