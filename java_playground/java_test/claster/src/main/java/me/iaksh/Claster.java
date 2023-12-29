package me.iaksh;

import org.lwjgl.openal.*;

public class Claster {
    private ALCCapabilities alcCapabilities;
    private ALCapabilities alCapabilities;
    private long device;
    private long context;
    private long source;
    private long pcm;

    private void genPCM(int freq) {
        int sampleRate = 44100;
        int samplesPerCycle = sampleRate / freq;
        short[] data = new short[10 * samplesPerCycle];
        for (int i = 0; i < data.length; i++) {
            if (i % samplesPerCycle < samplesPerCycle / 2) {
                data[i] = 0b00000000;
            }
            else {
                data[i] = 0b11111111;
            }
        }
        // this is bad, I need DMA
        if(pcm != 0) {
            AL11.alDeleteBuffers((int) pcm);
            pcm = AL11.alGenBuffers();
        }
        AL11.alBufferData((int) pcm, AL11.AL_FORMAT_MONO8, data, sampleRate);
    }

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
        pcm = AL11.alGenBuffers();
    }

    public Claster() {
        initOpenAL();
    }

    public void play(int freq,long ms) throws InterruptedException {
        genPCM(freq);
        AL11.alSourcei((int) source,AL11.AL_BUFFER,(int) pcm);
        AL11.alSourcei((int) source,AL11.AL_LOOPING,1);
        AL11.alSourcePlay((int) source);

        Thread.sleep(ms);
        AL11.alSourceStop((int) source);
    }

    public void destroy() {
        ALC11.alcDestroyContext(context);
        ALC11.alcCloseDevice(device);
    }
}
