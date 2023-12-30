package me.iaksh.cluster;

import org.lwjgl.openal.AL11;

public class OpenALBuffer {
    private int buffer;

    public OpenALBuffer() {
        buffer = AL11.alGenBuffers();
    }

    public void write(short[] data,int sampleRate) {
        AL11.alDeleteBuffers(buffer);
        buffer = AL11.alGenBuffers();
        AL11.alBufferData(buffer,AL11.AL_FORMAT_MONO16,data,sampleRate);
    }

    public int getBufferID() {
        return buffer;
    }

    public void destroyOpenALBuffer() {
        AL11.alDeleteBuffers(buffer);
    }
}
