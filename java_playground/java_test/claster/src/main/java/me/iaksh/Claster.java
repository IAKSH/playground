package me.iaksh;

import org.lwjgl.openal.*;

public class Claster {
    private int alSource;
    private boolean playing;

    void initAlSource() {
        alSource = AL11.alGenSources();
        AL11.alSourcei(alSource,AL11.AL_LOOPING,1);
        AL11.alSourcef(alSource,AL11.AL_GAIN,1.0f);
    }

    public Claster() {
        playing = false;
        initAlSource();
    }

    public void bindWave(Wave wave) {
        stop();
        AL11.alSourcei(alSource,AL11.AL_BUFFER,wave.getAlBuffer());
    }

    public void pause() {
        playing = false;
        AL11.alSourcePause(alSource);
    }

    public void resume() {
        // TODO
        playing = true;
    }

    public void stop() {
        playing = false;
        AL11.alSourceStop(alSource);
    }

    public void start() {
        playing = true;
        AL11.alSourcePlay(alSource);
    }

    public void setGain(float gain) {
        if(gain < 0.0f)
            throw new IllegalArgumentException();
        AL11.alSourcef(alSource,AL11.AL_GAIN,gain);
    }

    public float getGain() {
        return AL11.alGetSourcef(alSource,AL11.AL_GAIN);
    }

    public void destroyAlSource() {
        AL11.alDeleteSources(alSource);
    }

    public boolean isPlaying() {
        return playing;
    }
}
