package me.iaksh.player;

import me.iaksh.oscillator.Oscillator;
import org.lwjgl.openal.AL11;

public class Player {

    private static OpenALLoader loader;
    private int source;
    private int buffer;

    private void tryLoadOpenAL() {
        if(loader == null)
            loader = new OpenALLoader();
    }

    private void initSourceAndBuffer() {
        source = AL11.alGenSources();
        buffer = AL11.alGenBuffers();
    }

    private void waitTillSourceFinish() {
        try {
            while(AL11.alGetSourcei(source,AL11.AL_SOURCE_STATE) != AL11.AL_STOPPED)
                Thread.sleep(1);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public Player() {
        tryLoadOpenAL();
        initSourceAndBuffer();
    }

    public void play(float gain,short[] data) {
        AL11.alSourcef(source,AL11.AL_GAIN,gain);
        AL11.alBufferData(buffer,AL11.AL_FORMAT_MONO16,data,Oscillator.getSampleRate());
        AL11.alSourcei(source,AL11.AL_BUFFER,buffer);
        AL11.alSourcePlay(source);
        waitTillSourceFinish();
    }

    public static void closeOpenAL() {
        if(loader != null)
            loader.closeOpenAL();
        else
            throw new RuntimeException("OpenAL is not initialized, there's no need to close it");
    }
}
