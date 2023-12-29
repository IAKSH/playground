package me.iaksh;

import org.lwjgl.openal.*;

public class Claster {
    private ALCCapabilities alcCapabilities;
    private ALCapabilities alCapabilities;
    private long device;
    private long context;
    private int source;

    private void initOpenAL() {
        String defaultDeviceName = ALC11.alcGetString(0, ALC11.ALC_DEFAULT_DEVICE_SPECIFIER);
        device = ALC11.alcOpenDevice(defaultDeviceName);

        int[] attributes = {0};
        context = ALC11.alcCreateContext(device, attributes);
        ALC11.alcMakeContextCurrent(context);

        alcCapabilities = ALC.createCapabilities(device);
        alCapabilities = AL.createCapabilities(alcCapabilities);

        if(!(alCapabilities.OpenAL10 && alCapabilities.OpenAL11)) {
            System.out.println("OpenAL 10/11 is not supported in current context.");
        }

        source = AL11.alGenSources();
    }

    public Claster() {
        initOpenAL();
    }

    public void play(Wave wave) throws InterruptedException {
        AL11.alSourcei(source,AL11.AL_BUFFER,wave.getAlBuffer());
        AL11.alSourcei(source,AL11.AL_LOOPING,1);
        AL11.alSourcef(source,AL11.AL_GAIN,1.0f);
        AL11.alSourcePlay(source);

        Thread.sleep(wave.getDurationMs());
        AL11.alSourceStop(source);
    }

    public void destroy() {
        ALC11.alcDestroyContext(context);
        ALC11.alcCloseDevice(device);
    }
}
