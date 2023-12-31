package me.iaksh.mixer;

import org.lwjgl.openal.*;

public class OpenALLoader {
    private ALCCapabilities alcCapabilities;
    private ALCapabilities alCapabilities;
    private long device;
    private long context;

    private void loadOpenAL() {
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
    }

    public OpenALLoader() {
        loadOpenAL();
    }

    public void closeOpenAL() {
        ALC11.alcDestroyContext(context);
        ALC11.alcCloseDevice(device);
    }
}
